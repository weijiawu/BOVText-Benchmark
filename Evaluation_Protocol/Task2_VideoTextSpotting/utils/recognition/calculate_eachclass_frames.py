import os
import json
import cv2
from tqdm import tqdm
import numpy as np
import math
from utils import write_result_as_txt,debug, setup_logger,write_lines,MyEncoder
import re


import numpy as np
import math

#逆时针旋转
def Nrotate(angle,valuex,valuey,pointx,pointy):
    angle = (angle/180)*math.pi
    valuex = np.array(valuex)
    valuey = np.array(valuey)
    nRotatex = (valuex-pointx)*math.cos(angle) - (valuey-pointy)*math.sin(angle) + pointx
    nRotatey = (valuex-pointx)*math.sin(angle) + (valuey-pointy)*math.cos(angle) + pointy
    return (nRotatex, nRotatey)
#顺时针旋转
def Srotate(angle,valuex,valuey,pointx,pointy):
    angle = (angle/180)*math.pi
    valuex = np.array(valuex)
    valuey = np.array(valuey)
    sRotatex = (valuex-pointx)*math.cos(angle) + (valuey-pointy)*math.sin(angle) + pointx
    sRotatey = (valuey-pointy)*math.cos(angle) - (valuex-pointx)*math.sin(angle) + pointy
    return (sRotatex,sRotatey)
#将四个点做映射
def rotatecordiate(angle,rectboxs,pointx,pointy):
    output = []
    for rectbox in rectboxs:
        if angle>0:
            output.append(Srotate(angle,rectbox[0],rectbox[1],pointx,pointy))
        else:
            output.append(Nrotate(-angle,rectbox[0],rectbox[1],pointx,pointy))
    return output

def get_patch(image,contours):
    rect = cv2.minAreaRect(contours)#rect为[(旋转中心x坐标，旋转中心y坐标)，(矩形长，矩形宽),旋转角度]
    box_origin = cv2.boxPoints(rect)#box_origin为[(x0,y0),(x1,y1),(x2,y2),(x3,y3)]
    
    
    M = cv2.getRotationMatrix2D(rect[0],rect[2],1)
    dst = cv2.warpAffine(image,M,(2*image.shape[0],2*image.shape[1]))
    
    
    box = rotatecordiate(rect[2],box_origin,rect[0][0],rect[0][1])
    
    xs = [int(x[1]) for x in box]
    ys = [int(x[0]) for x in box]
#     print(dst.shape)
#     print(xs)
#     print(ys)
    cropimage = dst[min(xs):max(xs),min(ys):max(ys)]
    
    return cropimage

def get_annotation(video_path):
    annotation = {}
    with open(video_path,'r',encoding='utf-8-sig') as load_f:
        gt = json.load(load_f)

    for child in gt:
        lines = gt[child]
        annotation.update({child:lines})
    return annotation

train_data_dir = '/share/wuweijia/Data/MMVText/train'
train_list = os.path.join(train_data_dir,"train_list.txt")
image_path = os.path.join(train_data_dir,"image")
ann = os.path.join(train_data_dir,"annotation")
output_root = os.path.join("/share/wuweijia/Data/MMVText/test/recognition/images")
idxx = 0
lines_gt = []

video_english = 0
video_english_ = 0
video_scene = 0
video_scene_ = 0

videolist = {}
for i in os.listdir(ann):
    
    cls = os.path.join(ann,i)
    cls_num = 0
    for video in os.listdir(cls):
        ann_one = os.path.join(cls,video)
        
        annotation = get_annotation(ann_one)
        for frame_id in annotation.keys():

            annotatation_frame = annotation[frame_id]
            cls_num+=len(annotatation_frame)
                  
    videolist[i] = cls_num


print(videolist)


    
