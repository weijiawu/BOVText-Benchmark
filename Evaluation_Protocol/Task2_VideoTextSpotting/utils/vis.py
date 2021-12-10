# -*- coding: utf-8 -*-
import cv2
import os
import copy
import numpy as np
import math
# import Levenshtein
from cv2 import VideoWriter, VideoWriter_fourcc
import json
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

import shutil

def Frames2Video(frames_dir=""):
    '''  将frames_dir下面的所有视频帧合成一个视频 '''
    img_root = frames_dir      #'E:\\KSText\\videos_frames\\video_14_6'
    image = cv2.imread(os.path.join(img_root,"1.jpg"))
    h,w,_ = image.shape

    out_root = frames_dir+".avi"
    # Edit each frame's appearing time!
    fps = 20
    fourcc = VideoWriter_fourcc(*"MJPG")  # 支持jpg
    videoWriter = cv2.VideoWriter(out_root, fourcc, fps, (w, h))
    im_names = os.listdir(img_root)
    num_frames = len(im_names)
    print(len(im_names))
    for im_name in tqdm(range(1, num_frames+1)):
        string = os.path.join( img_root, str(im_name) + '.jpg')
#         print(string)
        frame = cv2.imread(string)
        # frame = cv2.resize(frame, (w, h))
        videoWriter.write(frame)

    videoWriter.release()
    shutil.rmtree(img_root)
    
def get_annotation(video_path):
    annotation = {}

    

    with open(video_path,'r',encoding='utf-8-sig') as load_f:
        gt = json.load(load_f)

    for child in gt:
        lines = gt[child]
        annotation.update({child:lines})

    return annotation

def cv2AddChineseText(img, text, position, textColor=(0, 255, 0), textSize=30):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "./simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text(position, text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    
if __name__ == "__main__":
    root = "/share/wuweijia/Data/MMVText/train"
    
    result_path = "./show"
    
    image_path = os.path.join(root,"image")
    annotation_path = os.path.join(root,"annotation")
    
    is_annotation = False
    # "video_6_3" Cls1_ZYDS_Frames
    # ["Cls1_ZYDS","Cls2_ECY","Cls3_TY","Cls4_MRMX","Cls5_PP"]
    seqs = [        
#         "Cls1_ZYDS","Cls2_ECY","Cls3_TY","Cls4_MRMX","Cls5_PP"
#         "Cls6_XW","Cls7_YX","Cls8_XJ","Cls9_XYH","Cls10_YL"
#         "Cls11_YS","Cls12_QG","Cls13_FCJJ","Cls14_CY","Cls15_SY"
#         "Cls16_ZWH","Cls17_JY","Cls18_LX","Cls19_SS","Cls20_XY"
        "Cls21_MY","Cls22_QC","Cls23_HW","Cls24_YY","Cls25_DJ"
#         "Cls26_KJ","Cls27_KP","Cls28_MY","Cls29_MZ","Cls30_WD"
           ]
#     seqs = ["Cls1_ZYDS_GtTxtsR3Frames"]
    
    for idx,cls in enumerate(os.listdir(annotation_path)):
        video_path = os.path.join(annotation_path, cls)
        video_frame = os.path.join(image_path, cls.replace("GtTxtsR4Frames","Frames"))
        
        if cls.replace("_GtTxtsR4Frames","") not in seqs:
            continue
            
        result_path_cls = os.path.join(result_path, cls.replace("GtTxtsR4Frames","Frames"))
        if not os.path.exists(result_path_cls):
            os.makedirs(result_path_cls)
            
        for video_name in tqdm(os.listdir(video_path)):
            annotation_path_ = os.path.join(video_path, video_name)
            
            video_path_ = os.path.join(video_frame, video_name.split(".json")[0])
            annotation = get_annotation(annotation_path_)
            
            result_path_cls_video = os.path.join(result_path_cls, video_name.split(".json")[0])
            if not os.path.exists(result_path_cls_video):
                os.makedirs(result_path_cls_video)
 
                
            for frame_id in annotation.keys():
                frame_name = video_name.split(".json")[0] + "_" + frame_id.zfill(6) + ".jpg"
                frame_path = os.path.join(video_path_,frame_name)
                frame = cv2.imread(frame_path)
#                 print(frame_path)
                
                
                annotatation_frame = annotation[frame_id]
                for data in annotatation_frame:
                    x1,y1,x2,y2,x3,y3,x4,y4,content,is_caption,ID = data
#                     print(data)
                    id_content = str(content) + str(ID)
#                     print(id_content)
#                     print(frame.shape)

    
                    if is_caption == "scene":
                        points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)
                        cv2.polylines(frame, [points], True, (0, 0, 255), thickness=3)
                        frame=cv2AddChineseText(frame,id_content, (int(x1), int(y1) - 20),(0, 255, 0), 45)
                    else:
                        points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)
                        cv2.polylines(frame, [points], True, (0, 255, 0), thickness=3)
                        frame=cv2AddChineseText(frame,id_content, (int(x1), int(y1) - 20),(0, 255, 0), 45)
                        
#                 if not os.path.exists(result_path):
#                     os.makedirs(result_path)
                frame_vis_path = os.path.join(result_path_cls_video, frame_id+".jpg")
                cv2.imwrite(frame_vis_path, frame)
#             video_vis_path = "./"
            Frames2Video(frames_dir=result_path_cls_video)
#             break
#         break
            
        
        
      
        
        