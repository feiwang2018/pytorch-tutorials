#Author:wangfei
#Datetime:2019/5/18 17:40
import torch
x=torch.tensor([[2.0,3.0,4.0],[4.0,5.0,6.0]],requires_grad=True)
y=x*x
print('x size',x.size()[1:])
print(y)
gradient=torch.tensor(torch.ones_like(y))
y.backward(gradient)
print('x grad is ',x.grad)
#help(y.backward)
#help(torch.nn.Conv2d)
size=x.size()[0:]
print("size is",size)
tol=1
for s in size:
    tol *=s
print(tol)
import matplotlib.pyplot as plt
# plt.ico()
import torchvision
help(torchvision.datasets.DatasetFolder)