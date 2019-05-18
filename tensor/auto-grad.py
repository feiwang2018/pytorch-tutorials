#Author:wangfei
#Datetime:2019/5/18 15:38
from __future__ import print_function
import torch
x=torch.ones(3,4,requires_grad=True)#创建tensor 并将requires_grad 设置为true开始计算梯度
print(x)
#需要求梯度的tensor需将requires_grad=True,当计算完成后，可以调用.backward()来自动计算所有tensor的梯度，
# 所有计算好的梯度放在.grad属性中,为了停止梯度求解，可以调用.detach()函数，为了保护梯度，将其放在with torch.no_grad()代码块中，多用于
# 测试模型，以为此时模型参数需要设置成requires_grad(),该参数默认为false，但是又不需要求解梯度
#一个tensor与参数设置成requires_grad=True的tensor相加时，构成tensor函数
print(x.grad)
y=x+2#
print(y.grad)
print(y)
#
print(y.grad_fn)
z=y*y*3
out=z.mean()
print(z)
print(out)
print(z,out)



#### 设置requires_grad的另一种方法,通过x.requires_grad_(True)设置
aa=torch.rand(3,3)
out=aa*3
print(aa.requires_grad)
aaa=aa.requires_grad_(True)
print(aaa.requires_grad)
# out.backward()
print(aa.grad)
print(aaa.grad)




#print(aaa.backward(torch.tensor([0.1,0.1])))#当需求解的y为标量时，直接y.backward(),当不是标量，
# 则需要给backward（）里添加y的维数,参考https://juejin.im/post/5b9b7a8cf265da0af1612bbd
# 梯度求解
y1=torch.tensor([[2.,3.,4.],[5.,2.,6.]],requires_grad=True)#梯度求导仅支持浮点数，
print(y1)
y2=y1*y1+2
out=y2.mean()#当求导只能是标量对标量求导，标量对向量或者矩阵求导，直接调用.backward()
gradients=torch.tensor(torch.ones_like(y2))
y2.backward(gradients)
print(y1.grad)#输出相应的梯度
# out.backward()
# print(y1.grad)#获得求得的导数


###使用  with torch.no_grad(): 停止求梯度
