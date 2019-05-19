#Author:wangfei
#Datetime:2019/5/19 15:26
#####  transfer learning
from __future__ import  print_function,division# 其中division项表示，导入python未来支持的精确除法，python2.x执行的为截断除法，如3/4=0,
#而导入精确除法后，3/4=0.75
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets,models,transforms
import matplotlib.pyplot as plt
import time
import os
import copy
###########  load data
##########  训练数据进行数据增强和标准化处理，验证数据集只进行标准化处理，通过定义字典来处理
