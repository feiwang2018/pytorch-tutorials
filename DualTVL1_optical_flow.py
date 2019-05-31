#Author:wangfei
#Datetime:2019/5/20 9:09
import cv2
import numpy as np
import time
from numba import jit,vectorize,cfunc,float32,cuda
import torch
#创建光流实例
# optflow=cv2.optflow.DualTVL1OpticalFlow_create()
#设置停止准则  epsilon的值  获取该值
# cv2.optflow.DualTVL1OpticalFlow_create.getEpsilon()
# optflow.getEpsilon()
#计算稠密光流
# help(optflow.calc)




# # import skvideo#  scikit-video  视频库读写类
cap=cv2.VideoCapture(r'E:\work2018\data\cholec80_1_.mp4')


ret,frame0=cap.read()
prvs=cv2.cvtColor(frame0,cv2.COLOR_BGR2GRAY)
# prvs=np.array(prvs)
# prvs=torch.from_numpy(prvs)
hsv=np.zeros_like(frame0)
hsv[...,1]=255
# @jit(nopython=True)
# help(jit)
# @vectorize(["float32(float32,float32)"],target='cuda')
# @cfunc("float32(float32,float32)")
# def calc_optical(prvs1,nextframe1):
#     # flow=None
#     print(prvs1.dtype,nextframe1.dtype)
#     print('prvs1 is',prvs1.shape)
#     #create_tvl1 = cv2.calcOpticalFlowFarneback(prvs, nextframe, None, 0.5, 3, 15, 3, 5, 1.2, 0)
#     create_tvl1=cv2.optflow.DualTVL1OpticalFlow_create()
#     flow=create_tvl1.calc(prvs1,nextframe1,None)
#     print(flow.dtype)#  float32
#     print(flow.shape)
#     return flow



while(1):
    ret,frame1=cap.read()
    nextframe=cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    # nextframe=np.array(nextframe )
    # nextframe=torch.from_numpy(nextframe)
    starttime=time.time()


    dualtvl1_opticalflow=cv2.optflow.DualTVL1OpticalFlow_create()
    #

#    @jit
#     flowDTVL1=dualtvl1_opticalflow.calc(prvs,nextframe,None)
    flowDTVL1 = dualtvl1_opticalflow.calc(prvs, nextframe,None)
#     help(dualtvl1_opticalflow.calc)
#     print(prvs.dtype,nextframe.dtype)
#     flowDTVL1=calc_optical(prvs,nextframe,None)
    print(flowDTVL1.dtype)
    # flowDTVL1=cv2.calcOpticalFlowFarneback(prvs,nextframe,None,0.5,3,15,3,5,1.2,0)endtime=time.time()
    endtime = time.time()
    #print(flowDTVL1.shape)
    mag,ang=cv2.cartToPolar(flowDTVL1[...,0],flowDTVL1[...,1])
    hsv[...,0]=ang*180/np.pi/2
    hsv[...,2]=cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    # endtime=time.time()
    fps=1/(endtime-starttime)#计算帧率
    print("fps is",fps)
    rgb=cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    cv2.imshow('opticalflow image',rgb)

    cv2.imshow('original image',frame1)
    k=cv2.waitKey(30) & 0xff
    if k==27:
        break
    elif k==ord('s'):
        cv2.imwrite('opt',frame1)
        cv2.imwrite('rgb',rgb)
    prvs=nextframe
cap.release()
cv2.destroyAllWindows()
