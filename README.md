# Action-Recognition
用ResNet做基于骨架的动作识别

## 下载数据集
下载[NTU RGB+D的骨架数据](https://github.com/shahroudy/NTURGB-D)

解压到 ./data/ntu/nturgb+d_skeletons/

## 数据预处理

```
 cd ./data/ntu

 python get_raw_skes_data.py

 python get_raw_denoised_data.py

 python seq_transformation.py
```

## 训练模型

```
python  main.py --aug 1 --train 1
```
