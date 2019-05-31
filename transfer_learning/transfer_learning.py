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

data_transfer={
    'train':transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
]),
    'val':transforms.Compose([transforms.Resize(256),
                              transforms.CenterCrop(224),
                              transforms.ToTensor(),
                              transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])}
data_dir='E:\\related-code\\pytorch-tutorials\\data\\hymenoptera_data\\hymenoptera_data'
image_datasets={x: datasets.ImageFolder(os.path.join(data_dir,x),data_transfer[x]) for x in ['train','val']}
dataloaders={x: torch.utils.data.DataLoader(image_datasets[x],batch_size=4,shuffle=True,num_workers=4) for x in ['train','val']}
dataset_sizes={x:len(image_datasets[x]) for x in ['train','val']}
class_name=image_datasets['train'].classes   #  return ant,bees
print(class_name)
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#########fining convnet 时，所有参数均被更新，pretrain model作为固定的特征提取器时，漆面的层被冻结，只有分类器层的参数被更新将


#  train model  and save best model
def train_model(model,criterion,optimizer,scheduler,num_epoches=25):
    start=time.time()
    best_model_wts=copy.deepcopy(model.state_dict())
    best_acc=0.0
    for epoch in range(num_epoches):
        print('epoch {}/{}'.format(epoch,num_epoches-1))


        for phase in ['train','val']:#  each epoch has training and val
            if phase=='train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss=0.0
            running_corrects=0


            #iterate over data
            for inputs,labels in dataloaders[phase]:
                inputs=inputs.to(device)# put it into gpu
                labels=labels.to(device)


                # zero param gradient
                optimizer.zero_grad()


                # forward ,track history if only in train
                with torch.set_grad_enabled(phase=='train'):
                    outputs=model(inputs)
                    _,preds=torch.max(outputs,1)
                    loss=criterion(outputs,labels)



                    #  backward if in train
                    if phase=='train':
                        loss.backward()
                        optimizer.step()


                    #  calc statistics
                    running_loss +=loss.item()*inputs.size(0)
                    running_corrects +=torch.sum(preds==labels.data)
            epoch_loss=running_loss/dataset_sizes[phase]
            epoch_acc=running_corrects.double()/dataset_sizes[phase]


            print('{} Loss is :{:.4f}  Acc is :{:4.f}'.format(phase,epoch_loss,epoch_acc))


            if phase=='val' and epoch_acc > best_acc:
                best_acc=epoch_acc
                best_model_wts=copy.deepcopy(model.state_dict())


    #  calc time all epoches
    time_elapsed=time.time()-start
    print('train time is {:.0f}m{:.0f}s'.format(time_elapsed//60,time_elapsed%60))
    print('best val acc:{:4f}'.format(best_acc))


    #  load best model weights
    model.load_state_dict(best_model_wts)
    return model



# load pretrained model
model_ft=models.resnet18(pretrained=True)
print(model_ft)# output model
num_fc=model_ft.fc.in_features#  get pretrained model's fc num
model_ft=model_ft.to(device)
criterion=nn.CrossEntropyLoss()
optimizer_ft=optim.SGD(model_ft.parameters(),lr=0.001,momentum=0.9)
#  decay lr by a factor 0.1 every 7 epoch
exp_lr_scheduler=lr_scheduler.StepLR(optimizer_ft,step_size=7,gamma=0.1)



#train and evaluate
model_ft=train_model(model_ft,criterion,optimizer_ft,exp_lr_scheduler,num_epoches=25)


######################################################################################

#pretrained as fixed feature
# model_ffe=models.resnet18(pretrained=True)
# for param in model_ffe.parameters():
#     param.requires_grad=False  #not update weights
# # new module need update gradient,so requires_grad=True
# num_ffe_fc=model_ffe.fc.in_features
# model_ffe.fc=nn.Linear(num_ffe_fc,2)#add new linear layer
# model_ffe=model_ffe.to(device)
# criterion=nn.CrossEntropyLoss()
# #  only final layer are being optimized
# optimizer_ffe=optim.SGD(model_ffe.fc.parameters(),lr=0.001,momentum=0.9)
# epr_lr_ffe_scheduler=lr_scheduler.StepLR(optimizer_ffe,step_size=7,gamma=0.1)
# ####### train and val
# model_ffe=train_model(model_ffe,criterion,epr_lr_ffe_scheduler,epr_lr_ffe_scheduler,num_epoches=25)





