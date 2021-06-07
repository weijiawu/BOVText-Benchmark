# -*- coding: utf-8 -*-
import cv2
import os
import copy
import numpy as np
import math
import Levenshtein
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
    annotation_path = "./results"
    image_path = '/share/wuweijia/Data/MMVText/train/image'
    result_root = '/home/guoxiaofeng/.jupyter/wuweijia/VideoTextSpotting/task123/MMVText'
    vis_path = "./vis"
    is_annotation = False
    # "video_6_3"
    seqs = ["Cls11_YS_Frames_2024021742.json","Cls11_YS_Frames_40595224195.json","Cls6_XW_Frames_40685610114.json" ,"Cls8_XJ_Frames_40159362262.json", "Cls8_XJ_Frames_41173254681.json"]
    
    for idx,video_path in enumerate(seqs):
        
            
        annotation_path_ = os.path.join(result_root,video_path)
        annotation = get_annotation(annotation_path_)
        
        name = video_path.replace("Frames","Frames")
        name = name.split("_")[0]+"_"+name.split("_")[1] + "_" +name.split("_")[2] + "/" + name.split("_")[3].split(".json")[0]

        video_path_ = os.path.join(image_path,name)
        video_vis_path = os.path.join(vis_path,video_path.split(".json")[0])
        
        print("Data preparing...{}".format(video_path))

        for image in tqdm(os.listdir(video_path_)):
            frame_path = os.path.join(video_path_,image)
            frame = cv2.imread(frame_path)
            
            hh = image.split(".jpg")[0].split("_")[-1]
            if str(int(hh)) in annotation.keys():
                annotatation_frame = annotation[str(int(hh))]

                for data in annotatation_frame:
                    if is_annotation:
                        x1,y1,x2,y2,x3,y3,x4,y4,ID,content = data
                        id_content = str(content)
                        if id_content == "###":
                            points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)
                            cv2.polylines(frame, [points], True, (0, 0, 255), thickness=3)
                            frame=cv2AddChineseText(frame,id_content, (int(x1), int(y1) - 20),(0, 255, 0), 35)
                        else:
                            points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)
                            cv2.polylines(frame, [points], True, (0, 255, 0), thickness=3)
                            frame=cv2AddChineseText(frame,id_content, (int(x1), int(y1) - 20),(0, 255, 0), 35)
    #                     cv2.putText(frame, id_content, (int(x1), int(y1) - 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.5,
    #                                 color=(0, 255, 0), thickness=2)
                    else:
                        x1,y1,x2,y2,x3,y3,x4,y4,ID = data
                        points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)
                        cv2.polylines(frame, [points], True, (0, 0, 255), thickness=3)
#                         frame=cv2AddChineseText(frame,id_content, (int(x1), int(y1) - 20),(0, 255, 0), 35)
            
                
            if not os.path.exists(video_vis_path):
                os.makedirs(video_vis_path)
            frame_vis_path = os.path.join(video_vis_path, str(int(hh))+".jpg")
            cv2.imwrite(frame_vis_path, frame)
            
        Frames2Video(frames_dir=video_vis_path)
        
        
        
        