#Author:wangfei
#Datetime:2019/5/18 21:47
#Author:wangfei
#Datetime:2019/5/18 20:49
#  目标：通过单隐层的来拟合随机数据  loss使用最小化欧氏距离
from __future__ import print_function
import torch


N,D_in,H,D_out=64,1000,100,10
x=torch.randn(N,D_in)
y=torch.randn(N,D_out)
model=torch.nn.Sequential(
    torch.nn.Linear(D_in,H),
    torch.nn.ReLU(),
    torch.nn.Linear(H,D_out),

)
loss_fn=torch.nn.MSELoss(reduction='sum')
learning_rate=1e-4
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
for i in range(500):
    y_pred=model(x)
    loss=loss_fn(y_pred,y)
    print("iter num is:",i,"loss is :",loss.item())
    # model.zero_grad()
    # loss.backward()
    # with torch.no_grad():
    #     for parm in model.parameters():
    #         parm -= learning_rate*parm.grad

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()