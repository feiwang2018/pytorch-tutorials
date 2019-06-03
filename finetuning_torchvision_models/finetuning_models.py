#-*-coding:utf-8-*-
#Author:wangfei
#datatime:2019/6/2上午10:30

from __future__ import print_function
from __future__ import division
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy
from torchvision import datasets,models,transforms
print(models)
###################
#########  define param
data_dir="./data/hymenoptera_data/hymenoptera_data/train/ants/"
print(data_dir)
# [resnet,alexnet,vgg,squeezenet,densenet,inception]
model_name="squeezenet"
num_class=2
batch_size=8
num_epochs=15
feature_extract=True
