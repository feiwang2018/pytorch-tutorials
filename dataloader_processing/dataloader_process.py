#Author:wangfei
#Datetime:2019/5/18 20:35
######  pandas 库方便处理csv格式数据，scikit-image:image-io and transforms
from __future__ import print_function
import torch
import os
import pandas as pd
from skimage import io,transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,utils
landmark_frame=pd.read_csv("E:\\related-code\\pytorch-tutorials\\data\\faces\\faces\\face_landmarks.csv")
n=65#  定义获取第65个人（行）的的人脸注释
img_name=landmark_frame.iloc[n,0]#获取第n行的第一列数据
landmarks=landmark_frame.iloc[n,1:].values#获取第二列以及后续所有列的数据
landmarks=landmarks.astype('float').reshape(-1,2)#调整两列的形式
print('img_name is :',img_name)
print('landmarks shape :{}'.format(landmarks.shape))
print('first 2 landmarks is{}'.format(landmarks[:4]))
#    ###########
   ###############  torch.utils.data.Dataset 创建自己的数据集类应该继承该类
   #__len__ 可以返回len(dataset)
   #__getitem__  可以利用dataset[i] 来获取第i个样本
## 创建数据集类的标准为一个字典  {'image':image,'landmarks':landmarks}

class facelandmarksDataset(Dataset):#  继承Dataset类
    def __init__(self,csv_file,root_dir,transform=None):
        self.landmarks_frame=pd.read_csv(csv_file)
        self.root_dir=root_dir
        self.transform=transform
    def __len__(self):
        return len(self.landmarks_frame)
    def __getitem__(self, item):# item  为索引，能够返回某个样本
        img_name=os.path.join(self.root_dir,self.landmarks_frame.iloc[item,0])
        image=io.imread(img_name)
        landmarks=self.landmarks_frame.iloc[item,1:].values
        landmarks=landmarks.astype('float').reshape(-1,2)#  (x,y)
        sample={'image':image,'landmarks':landmarks}#将数据点存入字典
        if self.transform:
            sample=self.transform(sample)
        return sample


########## 创建数据集实例
face_dataset=facelandmarksDataset(csv_file='E:\\related-code\\pytorch-tutorials\\data\\faces\\faces\\face_landmarks.csv',root_dir='E:\\related-code\\pytorch-tutorials\\data\\faces\\faces')
fig=plt.figure()
for i in range(len(face_dataset)):
    sample=face_dataset[i]
    print(i)
    print(i,sample['image'].shape,sample['landmarks'].shape)


########## transform    如rescale，randomcrop，totensor  ，其中numpy数组图像为HxWXC,torch为CxHXW
######  torchvision  数据预处理


data_transform=transforms.Compose([transforms.RandomSizedCrop(224),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
                                   ])

