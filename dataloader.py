# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import h5py
import os.path as osp
import sys
import scipy.misc
from PIL import Image

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


class NTUDataset(Dataset):
    """
    NTU Skeleton Dataset.

    Args:
        x (list): Input dataset, each element in the list is an ndarray corresponding to
        a joints matrix of a skeleton sequence sample
        y (list): Action labels
    """

    def __init__(self, x, y):
        self.x = x
        self.y = np.array(y, dtype='int')

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return [self.x[index], int(self.y[index])]


def tree_traversal_one_ske(ske_joint, dataset):
    shape = ske_joint.shape
    ret = np.zeros((shape[0], 1 + 2 * (shape[1] - 1), 3))
    if dataset in ['NTU', 'NTU120']:
        sequence = [1, 20, 2, 3, 2, 20,
                    4, 5, 6, 7, 21, 7, 22, 7, 6, 5, 4, 20,
                    8, 9, 10, 11, 22, 11, 24, 11, 10, 9, 8, 20, 1, 0,
                    12, 13, 14, 15, 14, 13, 12, 0,
                    16, 17, 18, 19, 18, 17, 16, 0, 1]
    elif dataset == 'UTD_MHAD':
        sequence = [2, 1, 0, 1,
                    8, 9, 10, 11, 10, 9, 8, 1,
                    4, 5, 6, 7, 6, 5, 4, 1,
                    2, 3, 16, 17, 18, 19, 18, 17, 16, 3,
                    12, 13, 14, 15, 14, 13, 12, 3, 2]
    elif dataset == 'SBU':
        sequence = [2, 1, 0, 1,
                    6, 7, 8, 7, 6, 1,
                    3, 4, 5, 4, 3, 1,
                    2, 12, 13, 14, 13, 12, 2,
                    9, 10, 11, 10, 9, 2]
    elif dataset == 'MSR':
        sequence = [3, 2, 19, 2,
                    1, 8, 10, 12, 10, 8, 1, 2,
                    0, 7, 9, 11, 9, 7, 0, 2, 3, 6,
                    5, 14, 16, 18, 16, 14, 5, 6,
                    4, 13, 15, 17, 15, 13, 4, 6, 3]
    elif dataset == 'FLO':
        sequence = [2, 1, 0, 1,
                    6, 7, 8, 7, 6, 1,
                    3, 4, 5, 4, 3, 1, 2,
                    12, 13, 14, 13, 12, 2,
                    9, 10, 11, 10, 9, 2]
    ret[:, :] = ske_joint[:, sequence]
    return ret


def tree_traversal_two_ske(ske_joint, dataset):
    if dataset in ['NTU', 'NTU120']:
        ske1 = ske_joint[:, :25]
        ske2 = ske_joint[:, 25:]
    elif dataset == 'SBU':
        ske1 = ske_joint[:, :15]
        ske2 = ske_joint[:, 15:]
    ret1 = tree_traversal_one_ske(ske1, dataset)
    ret2 = tree_traversal_one_ske(ske2, dataset)
    ret = np.concatenate([ret1, ret2], axis=1)
    return ret


class NTUDataLoaders(object):
    def __init__(self, dataset='NTU', case=1, aug=0):
        self.dataset = dataset
        self.case = case
        self.aug = aug
        self.create_datasets()
        self.train_set = NTUDataset(self.train_X, self.train_Y)
        self.val_set = NTUDataset(self.val_X, self.val_Y)
        self.test_set = NTUDataset(self.test_X, self.test_Y)

    def get_train_loader(self, batch_size, num_workers):
        if self.aug == 1:
            return DataLoader(self.train_set, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              collate_fn=self.collate_fn_aug, pin_memory=True)
        else:
            return DataLoader(self.train_set, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              collate_fn=self.collate_fn, pin_memory=True)

    def get_val_loader(self, batch_size, num_workers):
        return DataLoader(self.val_set, batch_size=batch_size,
                          shuffle=False, num_workers=num_workers,
                          collate_fn=self.collate_fn, pin_memory=True)

    def get_test_loader(self, batch_size, num_workers):
        return DataLoader(self.test_set, batch_size=batch_size,
                          shuffle=False, num_workers=num_workers,
                          collate_fn=self.collate_fn, pin_memory=True)

    def tree_traversal(self, ske_joint):
        if self.dataset in ['NTU', 'NTU120']:
            shape = ske_joint.shape
            if shape[1] == 25:
                ret = tree_traversal_one_ske(ske_joint, self.dataset)
            else:
                ret = tree_traversal_two_ske(ske_joint, self.dataset)
        elif self.dataset in ['UTD_MHAD', 'MSR', 'FLO']:
            ret = tree_traversal_one_ske(ske_joint, self.dataset)
        elif self.dataset == 'SBU':
            ret = tree_traversal_two_ske(ske_joint, self.dataset)
        return ret

    def torgb(self, ske_joints):
        rgb = []
        maxmin = list()
        # mean = np.load('./data/ntu/NTU_' + self.metric + '_mean.npy').reshape(3, 1, 1)
        # std = np.load('./data/ntu/NTU_' + self.metric + '_std.npy').reshape(3, 1, 1)
        self.idx = 0
        for ske_joint in ske_joints:
            zero_row = []
            if self.dataset in ['NTU', 'NTU120']:
                for i in range(len(ske_joint)):
                    if (ske_joint[i, :] == np.zeros((1, 150))).all():
                        zero_row.append(i)
                ske_joint = np.delete(ske_joint, zero_row, axis=0)
                if (ske_joint[:, 0:75] == np.zeros((ske_joint.shape[0], 75))).all():
                    ske_joint = np.delete(ske_joint, range(75), axis=1)
                elif (ske_joint[:, 75:150] == np.zeros((ske_joint.shape[0], 75))).all():
                    ske_joint = np.delete(ske_joint, range(75, 150), axis=1)

            max_val = self.max
            min_val = self.min
            # max_velocity = self.max_velocity
            # min_velocity = self.min_velocity
            # max_acceleration = self.max_acceleration
            # min_acceleration = self.min_acceleration

            ske_joint = np.reshape(ske_joint,
                                   (ske_joint.shape[0], ske_joint.shape[1] // 3, 3))  # (time_step, num_joint, 3)
            ske_joint = self.tree_traversal(ske_joint)
            # velocity = ske_joint[1:] - ske_joint[:-1]
            # ske_joint = ske_joint[2:]
            # acceleration = velocity[1:] - velocity[:-1]
            # velocity = velocity[1:]

            #### original rescale to 0-255
            ske_joint = 255 * (ske_joint - min_val) / (max_val - min_val)
            # velocity = 255 * (velocity - min_velocity) / (max_velocity - min_velocity)
            # acceleration = 255 * (acceleration - min_acceleration) / (max_acceleration - min_acceleration)

            # rgb_ske = scipy.misc.imresize(rgb_ske, (224, 224)).astype(np.float32)
            rgb_ske = np.array(Image.fromarray(np.uint8(ske_joint)).resize((224, 224))).astype(np.float32)  # (224, 50, 3)
            rgb_ske = center(rgb_ske)  # (224, 50, 3)
            # rgb_velocity = np.array(Image.fromarray(np.uint8(velocity)).resize((224, 224))).astype(np.float32)
            # rgb_velocity = center(rgb_velocity)
            # rgb_acceleration = np.array(Image.fromarray(np.uint8(acceleration)).resize((224, 224))).astype(np.float32)
            # rgb_acceleration = center(rgb_acceleration)
            # rgb_ske = np.concatenate([rgb_ske, rgb_velocity, rgb_acceleration], axis=-1)
            # rgb_ske = np.pad(rgb_ske, pad_width=((0, 300 - rgb_ske.shape[0]), (0, 224 - rgb_ske.shape[1]), (0, 0)),
            #                  mode='constant', constant_values=0).astype(np.float32)  # (300, 224, 3)
            rgb_ske = np.transpose(rgb_ske, [1, 0, 2])  # (50, 224, 3)
            rgb_ske = np.transpose(rgb_ske, [2, 1, 0])  # (3, 224, 50)
            # rgb_ske = (rgb_ske - mean) / std
            rgb.append(rgb_ske)
            maxmin.append([max_val, min_val])
            self.idx = self.idx + 1

        return rgb, maxmin

    def compute_max_min(self, ske_joints):
        max_vals, min_vals = list(), list()
        for ske_joint in ske_joints:
            zero_row = []
            if self.dataset in ['NTU', 'NTU120']:
                for i in range(len(ske_joint)):
                    if (ske_joint[i, :] == np.zeros((1, 150))).all():
                        zero_row.append(i)
                ske_joint = np.delete(ske_joint, zero_row, axis=0)
                if (ske_joint[:, 0:75] == np.zeros((ske_joint.shape[0], 75))).all():
                    ske_joint = np.delete(ske_joint, range(75), axis=1)
                elif (ske_joint[:, 75:150] == np.zeros((ske_joint.shape[0], 75))).all():
                    ske_joint = np.delete(ske_joint, range(75, 150), axis=1)

            max_val = ske_joint.max()
            min_val = ske_joint.min()
            max_vals.append(float(max_val))
            min_vals.append(float(min_val))
        max_vals, min_vals = np.array(max_vals), np.array(min_vals)

        return max_vals.max(), min_vals.min()

    def collate_fn_aug(self, batch):
        x, y = zip(*batch)
        x = torch.stack([torch.from_numpy(x[i]) for i in range(len(x))], 0)
        x = _transform(x)
        x, maxmin = self.torgb(x.numpy())

        x = torch.stack([torch.from_numpy(x[i]) for i in range(len(x))], 0)
        y = torch.LongTensor(y)
        return [x, torch.FloatTensor(maxmin), y]

    def collate_fn(self, batch):
        x, y = zip(*batch)
        x, maxmin = self.torgb(x)
        x = torch.stack([torch.from_numpy(x[i]) for i in range(len(x))], 0)
        y = torch.LongTensor(y)
        return [x, torch.FloatTensor(maxmin), y]

    def get_train_size(self):
        return len(self.train_Y)

    def get_val_size(self):
        return len(self.val_Y)

    def get_test_size(self):
        return len(self.test_Y)

    def create_datasets(self):
        if self.dataset == 'NTU':
            if self.case == 0:
                self.metric = 'CS'
            else:
                self.metric = 'CV'
            path = osp.join('./data/ntu', 'NTU_' + self.metric + '.h5')
        elif self.dataset == 'NTU120':
            if self.case == 0:
                self.metric = 'CS'
            else:
                self.metric = 'CV'
            path = osp.join('./data/ntu120', 'NTU_' + self.metric + '.h5')
        elif self.dataset == 'UTD_MHAD':
            self.metric = 'CS'
            path = osp.join('./data/utd_mhad', 'UTD_MHAD_' + self.metric + '.h5')
        elif self.dataset == 'SBU':
            self.metric = str(self.case)
            path = osp.join('./data/sbu', 'SBU_' + self.metric + '.h5')
        elif self.dataset == 'MSR':
            self.metric = str(self.case)
            path = osp.join('./data/msr', 'MSR_AS' + self.metric + '.h5')
        elif self.dataset == 'FLO':
            self.metric = str(self.case)
            path = osp.join('./data/flo', 'FLO_' + self.metric + '.h5')

        f = h5py.File(path, 'r')
        self.train_X = f['x'][:]
        self.train_Y = np.argmax(f['y'][:], -1)
        self.val_X = f['test_x'][:]
        self.val_Y = np.argmax(f['test_y'][:], -1)
        self.test_X = f['test_x'][:]
        self.test_Y = np.argmax(f['test_y'][:], -1)

        if self.dataset in ['NTU', 'NTU120']:
            self.max = 5.18858098984
            self.min = -5.28981208801
            # self.max_velocity = 4.9251251220703125
            # self.min_velocity = -4.901503086090088
            # self.max_acceleration = 9.234665870666504
            # self.min_acceleration = -9.141777992248535
        else:
            x = np.concatenate([self.train_X, self.val_X, self.test_X], 0)
            max_val, min_val = self.compute_max_min(x)
            self.max = max_val
            self.min = min_val


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def center(rgb):
    rgb[:, :, 0] -= 110
    rgb[:, :, 1] -= 110
    rgb[:, :, 2] -= 110
    return rgb


def padding(joints, max_len=300, pad_value=0.):
    num_frames, feat_dim = joints.shape
    if feat_dim == 75:
        joints = np.hstack((joints, np.zeros((num_frames, 75), dtype=joints.dtype)))
    if num_frames < max_len:
        joints = np.vstack(
            (joints, np.ones((max_len - num_frames, 150), dtype=joints.dtype) * pad_value))

    return joints


def _rot(rot):
    cos_r, sin_r = rot.cos(), rot.sin()
    zeros = rot.new(rot.size()[:2] + (1,)).zero_()
    ones = rot.new(rot.size()[:2] + (1,)).fill_(1)

    r1 = torch.stack((ones, zeros, zeros), dim=-1)
    rx2 = torch.stack((zeros, cos_r[:, :, 0:1], sin_r[:, :, 0:1]), dim=-1)
    rx3 = torch.stack((zeros, -sin_r[:, :, 0:1], cos_r[:, :, 0:1]), dim=-1)
    rx = torch.cat((r1, rx2, rx3), dim=2)

    ry1 = torch.stack((cos_r[:, :, 1:2], zeros, -sin_r[:, :, 1:2]), dim=-1)
    r2 = torch.stack((zeros, ones, zeros), dim=-1)
    ry3 = torch.stack((sin_r[:, :, 1:2], zeros, cos_r[:, :, 1:2]), dim=-1)
    ry = torch.cat((ry1, r2, ry3), dim=2)

    rz1 = torch.stack((cos_r[:, :, 2:3], sin_r[:, :, 2:3], zeros), dim=-1)
    r3 = torch.stack((zeros, zeros, ones), dim=-1)
    rz2 = torch.stack((-sin_r[:, :, 2:3], cos_r[:, :, 2:3], zeros), dim=-1)
    rz = torch.cat((rz1, rz2, r3), dim=2)

    rot = rz.matmul(ry).matmul(rx)

    return rot


def _transform(x):
    x = x.contiguous().view(x.size()[:2] + (-1, 3))

    rot = x.new(x.size()[0], 3).uniform_(-0.3, 0.3)

    rot = rot.repeat(1, x.size()[1])
    rot = rot.contiguous().view((-1, x.size()[1], 3))
    rot = _rot(rot)
    x = torch.transpose(x, 2, 3)
    x = torch.matmul(rot, x)
    x = torch.transpose(x, 2, 3)

    x = x.contiguous().view(x.size()[:2] + (-1,))
    return x


def make_dir(dataset, case, subdir):
    # if dataset == 'NTU':
    #     output_dir = os.path.join('./models/va-cnn/NTU/')
    # elif dataset == 'UTD_MHAD':
    #     output_dir = os.path.join('./models/va-cnn/UTD_MHAD/')
    # elif dataset == 'SBU':
    #     output_dir = os.path.join('./models/va-cnn/SBU/')
    output_dir = os.path.join('./models/va-cnn/' + dataset + '/')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    return output_dir


def get_cases(dataset):
    if dataset in ['NTU', 'NTU120']:
        cases = 2
    elif dataset == 'UTD_MHAD':
        cases = 1
    elif dataset == 'SBU':
        cases = 5
    elif dataset == 'MSR':
        cases = 3
    elif dataset == 'FLO':
        cases = 10

    return cases


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def get_num_classes(dataset):
    if dataset == 'NTU':
        return 60
    elif dataset == 'NTU120':
        return 120
    elif dataset == 'UTD_MHAD':
        return 27
    elif dataset in ['SBU', 'MSR']:
        return 8
    elif dataset == 'FLO':
        return 9
