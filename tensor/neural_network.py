#Author:wangfei
#Datetime:2019/5/18 19:23
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

######  定义神经网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()#需要计算梯度
        self.conv1=nn.Conv2d(1,6,5)#in_channel,out_channel,kernel_size,
        self.conv2=nn.Conv2d(6,16,5)
        self.fc1=nn.Linear(16*5*5,84)#in_feature,out_feature
        self.fc2=nn.Linear(84,10)
    def forward(self,x):
        x=F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x=F.max_pool2d(F.relu(self.conv2(x)),(2,2))
        x=x.view(-1,self.num_flat_feature(x))#第二维为特征向量的维数
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        return x



    def num_flat_feature(self,x):
        size=x.size()[1:]#获取batch维以外的所有维数大小
        num_feature=1
        for s in size:
            num_feature *=s
        return num_feature
net=Net()
print(net)
# print("learned parm is",net.parameters())
parm=list(net.parameters())#返回模型的学习的参数
print("learned para",parm[0].size())


################## 创建输入训练数据
input=torch.randn(1,1,32,32)
out=net(input)
print("out is ",out)

#########将所有参数的梯度初始化为零
#net.zero_grad()
###### 返回梯度
#out.backward(torch.randn(1,10))#backward里的参数形状需要和out的维数形状一致
##########
########## 计算loss
### loss函数的输入形式为：out，target。
target=torch.randn(10)
target=target.view(1,-1)#第二维为output的shape
criterion=nn.MSELoss()
loss=criterion(out,target)
# print("loss is ",loss)
# print(loss.grad_fn)

######### 开始反向传播
#初始化loss梯度为零
# net.zero_grad()
# print("conv1.grad is",net.conv1.bias.grad)
# loss.backward()
# print("after backward,conv1 grad is",net.conv1.bias.grad)
  ##### 更新权重
# weight=weight - learning_rate*gradient
#设置学习率  python 代码实现如下
# learning_rate=0.001
# for f in net.parameters():
#     f.data.sub_(f.grad.data*learning_rate)


###导入优化器
import torch.optim as optim
#创建优化器
optimizer=optim.SGD(net.parameters(),lr=0.01)#第一个参数为网络权重
#将优化器梯度初始化为零
optimizer.zero_grad()
loss=criterion(out,target)
loss.backward()
optimizer.step()
print("final loss is ",loss)
