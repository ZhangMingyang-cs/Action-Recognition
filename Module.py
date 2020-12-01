#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math


# In[45]:


class Attention(nn.Module):
    """
        返回值：
            返回的不是attention权重，而是每个timestep乘以权重后相加得到的向量。
        输入:
            (batch_size, step_dim, dims_to_weight, features_dim)
    """

    def __init__(self, dims_to_weight, features_dim, bias=True):
        super(Attention, self).__init__()
        self.dims_to_weight = dims_to_weight
        self.features_dim = features_dim
        self.bias = bias

        self.latent_dim = 64
        self.eps = 1e-5

        self.weight1 = nn.Parameter(torch.Tensor(self.features_dim, self.latent_dim))
        self.weight2 = nn.Parameter(torch.Tensor(self.latent_dim, 1))

        if self.bias:
            self.b1 = nn.Parameter(torch.Tensor(self.latent_dim))
            self.b2 = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight1.size(1))
        self.weight1.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.weight2.size(1))
        self.weight2.data.uniform_(-stdv, stdv)
        if self.bias:
            # self.encode_bias.data.uniform_(-stdv, stdv)
            # self.decode_bias.data.uniform_(-stdv, stdv)
            nn.init.zeros_(self.b1)
            nn.init.zeros_(self.b2)

    def forward(self, x):
        eij = F.relu(torch.matmul(x, self.weight1))  # (batch_size, step_dim, dims_to_weight, latent_dim)
        if self.bias:
            eij = torch.add(eij, self.b1)
        eij = torch.matmul(eij, self.weight2)  # (batch_size, step_dim, dims_to_weight, 1)
        if self.bias:
            eij = torch.add(eij, self.b2)

        # RNN一般默认激活函数为tanh, 对attention来说激活函数差别不大，因为要做softmax
        eij = torch.tanh(eij)
        a = torch.exp(eij)  # (batch_size, step_dim, dims_to_weight, 1)

        # cast是做类型转换，keras计算时会检查类型，可能是因为用gpu的原因
        a = torch.div(a, (torch.sum(a, dim=2, keepdim=True) + self.eps))  # (batch_size, step_dim, dims_to_weight, 1)

        # 此时a.shape = (batch_size, step_dim, dims_to_weight, 1),
        # x.shape = (batch_size, step_dim, dims_to_weight, features_dim)
        weighted_input = torch.add(torch.mul(x, a), x)

        return weighted_input

    def extra_repr(self):
        return 'dims_to_weight={}, features_dim={}, bias={}'.format(
            self.dims_to_weight, self.features_dim, self.bias
        )


# In[46]:


class AutoEncoder(nn.Module):
    """
    输入：(3, 224, 50)
    输出：(3, 224, 224)
    """
    def __init__(self, in_features, latent_features, bias=True):
        super(AutoEncoder, self).__init__()
        self.in_features = in_features
        self.latent_features = latent_features
        self.weight = nn.Parameter(torch.Tensor(latent_features, in_features))
        if bias:
            self.encode_bias = nn.Parameter(torch.Tensor(latent_features))
            self.decode_bias = nn.Parameter(torch.Tensor(in_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.encode_bias is not None:
            # self.encode_bias.data.uniform_(-stdv, stdv)
            # self.decode_bias.data.uniform_(-stdv, stdv)
            nn.init.zeros_(self.encode_bias)
            nn.init.zeros_(self.decode_bias)

    def forward(self, input):
        encoded = F.relu(F.linear(input, self.weight, self.encode_bias))
        decoded = F.linear(encoded, torch.transpose(self.weight, 0, 1), self.decode_bias)
        return encoded, decoded

    def extra_repr(self):
        return 'in_features={}, latent_features={}, bias={}'.format(
            self.in_features, self.latent_features, self.encode_bias is not None and self.decode_bias is not None
        )


# In[47]:


class AttentionAE(nn.Module):
    def __init__(self, dims_to_weight, features_dim, latent_features):
        super(AttentionAE, self).__init__()
        self.att = Attention(dims_to_weight=dims_to_weight, features_dim=features_dim)
        # self.bn_att = nn.BatchNorm2d(features_dim)
        self.autoencoder = AutoEncoder(in_features=dims_to_weight, latent_features=latent_features)

    def forward(self, x):
        # (batch_size, 3, 224, 50)
        x = x.permute(0, 2, 3, 1)
        # (batch_size, 224, 50, 3)
        x = self.att(x)
        # (batch_size, 224, 50, 3)
        x = x.permute(0, 3, 1, 2)
        # (batch_size, 3, 224, 50)
        # x = self.bn_att(x)
        encoded, decoded = self.autoencoder(x)
        # (batch_size, 3, 224, 224)

        return encoded, decoded


class AttentionFCResNet(nn.Module):
    def __init__(self, num_classes, ae_path=None, resnet_path=None):
        super(AttentionFCResNet, self).__init__()
        self.num_classes = num_classes
        self.ae_path = ae_path
        self.resnet_path = resnet_path

        self.att = Attention(224, 3)
        # self.bn_att = nn.BatchNorm2d(9)
        self.encode = nn.Linear(224, 224)
        self.encode_relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(3)

        self.classifier = models.resnet50(pretrained=True)
        num_ftrs = self.classifier.fc.in_features
        self.classifier.fc = nn.Linear(num_ftrs, self.num_classes)
        # self.classifier = ResNetWithDropout()

        if self.ae_path is not None or self.resnet_path is not None:
            self.init_weights()

    def init_weights(self):
        print('initialize weight')
        if self.ae_path is not None:
            autoencoder = AttentionAE(224, 3, 224)
            autoencoder.load_state_dict(torch.load(self.ae_path)['state_dict'], strict=False)

            self.att.weight1.data = torch.from_numpy(autoencoder.att.weight1.detach().numpy())
            self.att.weight2.data = torch.from_numpy(autoencoder.att.weight2.detach().numpy())
            self.att.b1.data = torch.from_numpy(autoencoder.att.b1.detach().numpy())
            self.att.b2.data = torch.from_numpy(autoencoder.att.b2.detach().numpy())

            # self.bn_att.weight.data = torch.from_numpy(autoencoder.bn_att.weight.detach().numpy())
            # self.bn_att.bias.data = torch.from_numpy(autoencoder.bn_att.bias.detach().numpy())

            self.encode.weight.data = torch.from_numpy(autoencoder.autoencoder.weight.detach().numpy())
            self.encode.bias.data = torch.from_numpy(autoencoder.autoencoder.encode_bias.detach().numpy())

        if self.resnet_path is not None:
            self.classifier.load_state_dict(torch.load(self.resnet_path)['state_dict'], strict=False)

    def forward(self, x):
        # (batch_size, 3, 224, 50)
        x = x.permute(0, 2, 3, 1)
        # (batch_size, 224, 50, 3)
        x = self.att(x)
        # (batch_size, 224, 50, 3)
        x = x.permute(0, 3, 1, 2)
        # (batch_size, 3, 224, 50)
        # x = self.bn_att(x)
        x = self.encode(x)
        # (batch_size, 3, 224, 224)
        x = self.encode_relu(x)
        x = self.bn(x)

        x = self.classifier(x)

        return x

# In[5]:


class Regularization(torch.nn.Module):
    def __init__(self,model,weight_decay,p=2):
        """
        :param model 模型
        :param weight_decay:正则化参数
        :param p: 范数计算中的幂指数值，默认求2范数,
                  当p=0为L2正则化,p=1为L1正则化
        """
        super(Regularization, self).__init__()
        if weight_decay <= 0:
            print("param weight_decay can not <=0")
            exit(0)
        self.model=model
        self.weight_decay=weight_decay
        self.p=p
        self.weight_list=self.get_weight(model)
        self.weight_info(self.weight_list)

    def to(self,device):
        """
        指定运行模式
        :param device: cuda or cpu
        :return:
        """
        self.device=device
        super().to(device)
        return self

    def forward(self, model):
        self.weight_list=self.get_weight(model)#获得最新的权重
        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay, p=self.p)
        return reg_loss

    def get_weight(self,model):
        """
        获得模型的权重列表
        :param model:
        :return:
        """
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list

    def regularization_loss(self,weight_list, weight_decay, p=2):
        """
        计算张量范数
        :param weight_list:
        :param p: 范数计算中的幂指数值，默认求2范数
        :param weight_decay:
        :return:
        """
        # weight_decay=Variable(torch.FloatTensor([weight_decay]).to(self.device),requires_grad=True)
        # reg_loss=Variable(torch.FloatTensor([0.]).to(self.device),requires_grad=True)
        # weight_decay=torch.FloatTensor([weight_decay]).to(self.device)
        # reg_loss=torch.FloatTensor([0.]).to(self.device)
        reg_loss=0
        for name, w in weight_list:
            l2_reg = torch.norm(w, p=p)
            reg_loss = reg_loss + l2_reg

        reg_loss=weight_decay*reg_loss
        return reg_loss

    def weight_info(self,weight_list):
        """
        打印权重列表信息
        :param weight_list:
        :return:
        """
        print("---------------regularization weight---------------")
        for name ,w in weight_list:
            print(name)
        print("---------------------------------------------------")

# In[6]:


class ResNetWithDropout(nn.Module):
    def __init__(self, num_classes):
        super(ResNetWithDropout, self).__init__()
        self.num_classes = num_classes
        self.resnet = models.resnet50(pretrained=True)
        # self.resnet.conv1 = nn.Conv2d(9, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Dropout(0.5)
        self.fc1 = nn.Linear(num_ftrs, self.num_classes)

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc1(x)
        return x

# In[7]:


class FCResNet(nn.Module):
    def __init__(self, num_classes, autoencoder=None):
        super(FCResNet, self).__init__()
        self.num_classes = num_classes
        self.autoencoder = autoencoder

        self.encode = nn.Linear(224, 224)
        self.encode_relu = nn.ReLU()

        self.classifier = models.resnet50(pretrained=True)
        num_ftrs = self.classifier.fc.in_features
        self.classifier.fc = nn.Linear(num_ftrs, self.num_classes)
        # self.classifier = ResNetWithDropout()

        if self.autoencoder is not None:
            self.init_weights()
        self.autoencoder = None

    def init_weights(self):
        print('initialize weight')
        self.encode.weight.data = torch.from_numpy(self.autoencoder.weight.detach().numpy())
        self.encode.bias.data = torch.from_numpy(self.autoencoder.encode_bias.detach().numpy())

    def forward(self, x):
        # (batch_size, 3, 224, 50)
        x = self.encode(x)
        # (batch_size, 3, 224, 224)
        x = self.encode_relu(x)
        x = self.classifier(x)
        return x

# In[8]:


class AttentionResNet(nn.Module):
    def __init__(self, num_classes):
        super(AttentionResNet, self).__init__()
        self.num_classes = num_classes

        self.att = Attention(224, 9)
        self.bn = nn.BatchNorm2d(9)

        self.classifier = models.resnet50(pretrained=True)
        self.classifier.conv1 = nn.Conv2d(9, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = self.classifier.fc.in_features
        self.classifier.fc = nn.Linear(num_ftrs, self.num_classes)

    def forward(self, x):
        # (batch_size, 3, 224, 50)
        x = x.permute(0, 2, 3, 1)
        # (batch_size, 224, 50, 3)
        x = self.att(x)
        # (batch_size, 224, 50, 3)
        x = x.permute(0, 3, 1, 2)
        # (batch_size, 3, 224, 50)
        x = self.bn(x)

        x = self.classifier(x)

        return x

# In[9]:


# In[10]:


# In[11]:


# In[12]:


# In[53]:


# In[43]:


# In[55]:


# In[ ]:
