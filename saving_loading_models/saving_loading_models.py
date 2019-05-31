#Author:wangfei
#Datetime:2019/5/19 20:23
#   torch.save  torch.load   torch.nnModule.load_state_dict
#############  example  定义模型
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
class ModelClass(nn.Module):
    def __init__(self):
        super(ModelClass, self).__init__()
        self.conv1=nn.Conv2d(3,6,5)
        self.pool1=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(6,16,5)
        self.fc1=nn.Linear(16*5*5,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)
    def forward(self,x):
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))
        x=x.view(-1,16*5*5)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        return x
#初始化模型
mymodel=ModelClass()
# 初始化优化其参数
optimzer=optim.SGD(mymodel.parameters(),lr=0.001,momentum=0.9)
# 将模型参数输出
print("model param")
for param in mymodel.state_dict():
    print(param,mymodel.state_dict()[param].size())
#  print optim param
for var_name in optimzer.state_dict():
    print(var_name,optimzer.state_dict()[var_name])


#  save model
#####  first method
torch.save(mymodel,'E:\\work2018\\model_save.pt')

########### second method
# torch.save({'epoch':epoch,
#             'model_state_dict':mymodel.state_dict(),
#             'optimizer_state_dict':optimzer.state_dict(),
#             'loss':loss},PATH)
# print(torch.save())

#load model
#  first method
mymodel_load=torch.load('E:\\work2018\\model_save.pt')
print("load model",mymodel_load)
mymodel_load.eval()#  before running ,must use mymodel_load.eval() to dropout and
# batch normalization layers to evaluation mode

#  load model ,second method
# load_model=Model()
# optimizer=optimzerclass()
# checkpoint=torch.load(PATH)
# load_model.load_state_dict(checkpoint['model_state_dict'])
# optimzer.load_state_dict(checkpoint['optimizer'])
# epoch=checkpoint['epoch']
# loss=checkpoint['loss']
#
#
# model.eval()
# #or
# model.train()