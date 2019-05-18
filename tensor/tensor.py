#Author:wangfei
#Datetime:2019/5/18 10:56
from __future__ import print_function#确保python2.x中不加括号的的print在python3.x中正常使用,且必须放在开头
import torch

print(torch.__version__)

x=torch.empty(3,4)#创建3x4的空张量
x2=torch.rand(3,4)#创建3x4的随机tensor
x3=torch.zeros(3,4)#创建3x4的全零tensor
x4=torch.tensor([3,4,5])#直接从data构建tensor
x5=x4.new_ones(3,2)#依据x4的张量构建新张量
x6=torch.randn_like(x5,dtype=torch.float)#依据x5张量的大小构建随机数张量,需要指定数据类型
print(x6.size())#输出的为张量维数的元组，符合元祖操作


########  张量的加法，需要两个张量维数相同
print(x+x2)  #tensor直接相加
print(x,x2)
# add 2
print(torch.add(x,x2))
#add 3
result=torch.empty(3,4)
print(torch.add(x,x2,out=result))#需要先提供一个同维度的输出张量来存储张量相加的结果
# #add 4
# result.add

print(x6)


#tensor 可以向numpy数组一样操作

print(x2[:,1])
######  重新调整张量大小

print(x.view(2,6).size(),x.size(),x.view(-1,2).size())#将原来3x4的tensor调节成2x6,其中-1为自动推断出的维数
#### 使用item()获取tensor中的一个元素的数字
print(x2[1,2].item())#获取第一行第二列的数


#x.numpy()将tensor转换成numpy数组
print(x.numpy())


#torch.from_numpy(numpy arr) 将numpy数组转换成tensor
a=x.numpy()

print(torch.from_numpy(a))
#######
#将tensor转换成CUDA tensor
if torch.cuda.is_available():
    device=torch.device("cuda")#cuda 设备对象
    y=torch.ones_like(x,device=device)#按照x维数大小在cuda上创建张量
    b=torch.ones_like(x)
    bb=b.to(device)#方法2直接将tensor放在gpu上
    z=y+bb
    print("z is ",z)