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
output_root = os.path.join("/share/wuweijia/Data/MMVText/test/recognition/images")
idxx = 0
lines_gt = []

video_english = 0
video_english_ = 0
video_scene = 0
video_scene_ = 0

# video = {}
# for i in os.listdir(image_path):
    
#     cls = os.path.join(image_path,i)
# #     if os.path.isfile(path)
#     num = 0
#     for image in os.listdir(cls):
#         image_ = os.path.join(cls,image)
#         if os.path.isfile(image_):
#             continue
#         num+= len(os.listdir(image_))
                  
#     video[i] = num

# print(video)

cha = {}
for i in "abcdefghijklmnopqrstuvwxyz":
    cha[i] = 0
with open(train_list, encoding='utf-8', mode='r') as f:
    for line in tqdm(f.readlines()):
        params = line.strip().strip('\ufeff').strip('\xef\xbb\xbf')

        video_path = os.path.join(os.path.join(train_data_dir,"annotation"),params)
        annotation = get_annotation(video_path.replace("GtTxtsR2Frames","GtTxtsR4Frames"))

        for frame_id in annotation.keys():

#             if int(frame_id)%3!=0:
#                 continue

            frame_name = params.split("/")[1].split(".json")[0] + "_" + frame_id.zfill(6) + ".jpg"
            frame_path = os.path.join(params.replace("GtTxtsR2Frames","Frames").split(".json")[0],frame_name)
            frame_path = os.path.join(image_path,frame_path)
#             self.img_paths.append(frame_path)

#                     print(frame_path)
#             img = get_img(frame_path)  43  68
#             print(frame_path)
#             img = cv2.imread(frame_path)
#             h, w = img.shape[0:2]
            annotatation_frame = annotation[frame_id]

            bboxes = []
            text_tags = []
            flag = 1
            flag_ = 1
            for data in annotatation_frame:
                x1,y1,x2,y2,x3,y3,x4,y4,content,is_caption,ID = data
                
                # 判断是否是全英文：isalpha
                # 判断是否是全数字: isdigit
                # isalnum()
                
                for cc in content:
                    if cc in cha:
                        cha[cc]+=1
#                 zhPattern = re.compile(u'[\u4e00-\u9fa5]+')
#                 match = zhPattern.search(content)
# #                 print(is_caption)
#                 object_boxes = []

#                 object_boxes = [np.array([int(x1),int(y1)]),np.array([int(x2),int(y2)]),
#                                 np.array([int(x3),int(y3)]),np.array([int(x4),int(y4)])]
#                 points = np.array(object_boxes)
#                 x, y, w, h = cv2.boundingRect(points)
#                 p3=p2-p1
#                 p1=np.array([int(x1),int(y1)])
#                 p2=np.array([int(x2),int(y2)])
#                 p3=np.array([int(x3),int(y3)])
#                 p4=np.array([int(x4),int(y4)])
                
#                 dis = []
#                 p5 = p2-p1
#                 dis.append(math.hypot(p5[0],p5[1]))
                
#                 p5 = p3-p2
#                 dis.append(math.hypot(p5[0],p5[1]))
                
                
#                 p5 = p4-p3
#                 dis.append(math.hypot(p5[0],p5[1]))
                
#                 p5 = p1-p4
#                 dis.append(math.hypot(p5[0],p5[1]))
                
#                 w,h = np.array(dis).max(),np.array(dis).min()
#                 if is_caption=="scene" and flag:
#                     video_english+=1
#                     video_english_ += w/h
# # #                     flag = 0
# # #                     break
#                 elif is_caption=="caption" and flag_:
#                     video_scene+=1
#                     video_scene_ += w/h
#                     flag_ = 0
#                     break
#                 if match:
#                     video_english+=1
#                     break
    
#                 if content.encode('UTF-8').isalnum():
#                     video_english+=1
#                     video_scene_ += w/h
#                     break
#                 points = np.array([x1, y1, x2, y2, x3, y3, x4, y4], np.int)
#                 points = np.array(points).flatten()
#                 points = points.reshape((int(len(points)/2), 2))
#             break
print(cha)

#                         label = params[8]
#                         if label == '*' or label == '###':
#                             text_tags.append(True)
#                         else:
#                             text_tags.append(False)
#                 print(points)
#                 print(img.shape)
#                 crop_image = get_patch(img,points)
    
#                 h,w = crop_image.shape[:2]
            
#                 if h==0 or w==0:
#                     continue
#                 if h>w:
#                     crop_image=np.rot90(crop_image,-1)
                
#                 output = os.path.join(output_root,str(idxx)+'.jpg')
#                 idxx += 1
                
# #                 if idxx>400:
# #                     break
                
#                 line = output
#                 line += '\t'
#                 line += content
#                 line += "\n"
                
#                 lines_gt.append(line)
#                 cv2.imwrite(output,crop_image)
#             break
#         break
    
    
# gt_name = "/share/wuweijia/Data/MMVText/test/recognition/groundtruth.txt"
# write_lines(gt_name, lines_gt)
    
