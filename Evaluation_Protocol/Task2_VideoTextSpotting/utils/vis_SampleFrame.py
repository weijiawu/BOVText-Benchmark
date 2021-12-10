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

def Frames2Video(frames_dir=""):
    '''  将frames_dir下面的所有视频帧合成一个视频 '''
    img_root = frames_dir      #'E:\\KSText\\videos_frames\\video_14_6'
    image = cv2.imread(os.path.join(img_root,"1.jpg"))
    h,w,_ = image.shape

    out_root = frames_dir+".avi"
    # Edit each frame's appearing time!
    fps = 10
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
    
def get_annotation(video_path):
    annotation = []
    
    with open(video_path,"r") as f:
        lines = f.readlines()
    
    for line in lines:
        box = line.rstrip('\n').lstrip('\ufeff').split(':')
        annotation.append(box)

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
    root = "/share/wuweijia/Data/MMVText/train/"
    
    result_path = "./show_SampleFrame"
    
    image_path = os.path.join(root,"image")
    annotation_path = os.path.join(root,"Annotation_S")
    
    is_annotation = False
    # "video_6_3" Cls1_ZYDS_Frames
    seqs = [   "Cls1_ZYDS_Train"]
    
    for idx,cls in enumerate(os.listdir(annotation_path)):
        
        if cls not in seqs:
            continue
        
        for video_sample in os.listdir(os.path.join(annotation_path,cls)):

            video_path = os.path.join(annotation_path, cls,video_sample)
            video_frame = os.path.join(image_path, cls.replace("Train","Frames"))


            result_path_cls = os.path.join(result_path, cls.replace("Train","Frames"))
            if not os.path.exists(result_path_cls):
                os.makedirs(result_path_cls)

            for frame_name in tqdm(os.listdir(video_path)):
                annotation_path_ = os.path.join(video_path, frame_name)

                video_path_ = os.path.join(video_frame, video_sample)
                try:
                    annotation = get_annotation(annotation_path_)
                except:
                    print(annotation_path_)
                    continue
                result_path_cls_video = os.path.join(result_path_cls, video_sample)
                if not os.path.exists(result_path_cls_video):
                    os.makedirs(result_path_cls_video)
                
                result_path_cls_image = os.path.join(result_path_cls_video, frame_name.split(".txt")[0]+".jpg")
                frame_name = os.path.join(video_path_, frame_name.split(".txt")[0] + ".jpg")
#                 print(frame_name)
                frame = cv2.imread(frame_name)
                
    
                for data in annotation:
                    try:
                        x1,y1,x2,y2,x3,y3,x4,y4,content,is_caption = data
                    except:
                        x1,y1,x2,y2,x3,y3,x4,y4,content,is_caption = data[:10]
                        print(data)
                        print(annotation_path_)
#                         assert False
#                     print(data)
                    id_content = str(content)

                    if is_caption == "背景文字":
                        points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)
                        cv2.polylines(frame, [points], True, (0, 0, 255), thickness=3)
                        frame=cv2AddChineseText(frame,id_content, (int(x1), int(y1) - 20),(0, 255, 0), 35)
                    else:
                        points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)
                        cv2.polylines(frame, [points], True, (0, 255, 0), thickness=3)
                        frame=cv2AddChineseText(frame,id_content, (int(x1), int(y1) - 20),(0, 255, 0), 35)

                cv2.imwrite(result_path_cls_image, frame)
 
            
        
        
      
        
        