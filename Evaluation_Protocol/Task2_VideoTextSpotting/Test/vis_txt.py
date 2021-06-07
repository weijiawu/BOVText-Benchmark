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



def get_bboxes(data_polygt):
    bboxes = []
    text_tags = []
    content = []
    is_caption = []
    with open(data_polygt, encoding='utf-8', mode='r') as f:
        for line in f.readlines():
            params = line.strip().strip('\ufeff').strip('\xef\xbb\xbf').split(':')
            try:
                label = params[8]
                if label == '*' or label == '###':
                    text_tags.append(True)
                else:
                    text_tags.append(False)
            except:
                text_tags.append(True)
                print(params)
            try:
                box = list(map(float, params[:8]))
            except:
                print(params)
                continue
            bboxes.append(box)
            try:
                content.append(params[8])
            except:
                content.append("###")
            try:
                is_caption.append(params[9])
            except:
                caption = "背景文字"
                is_caption.append(caption)
                print(params)
    return bboxes, text_tags, content, is_caption

if __name__ == "__main__":
    annotation_path = "./results"
    image_path = "/share/ICDAR2021/VideoSpotting/SVTS/val/images"
    vis_path = "./vis"
    is_annotation = False
    # "video_6_3"
    seqs = [   "video_18_4"]
    
    
    train_data_dir = '/share/caiyuanqiang/VideoSet'
    train_gt_dir = '/share/wuweijia/Data/VideoSet(Ours)/Train/Annotation'
    output = "./show"
    for video_cls in os.listdir(train_gt_dir):
        video_cls_path = os.path.join(train_gt_dir,video_cls)
        if os.path.isfile(video_cls_path):
            continue
                
        output_vis = os.path.join(output,video_cls) 
        if not os.path.exists(output_vis):
            os.makedirs(output_vis)
                
        for video_path in os.listdir(video_cls_path):
            video_path_ = os.path.join(video_cls_path,video_path)
            
            output_vis1 = os.path.join(output_vis,video_path) 
            if not os.path.exists(output_vis1):
                os.makedirs(output_vis1)

            for idx1,frame_name in enumerate(os.listdir(video_path_)):
                gt_frame_path = os.path.join(video_path_,frame_name)
                if idx1>10:
                    break
                image_frame_path = os.path.join(train_data_dir,video_cls.replace("Train","SampleFrames"))
                image_frame_path = os.path.join(image_frame_path,video_path)
                image_frame_path = os.path.join(image_frame_path,frame_name.replace(".txt",".jpg"))
                
                frame = cv2.imread(image_frame_path)
                annotatation_frame, text_tags, contents, is_captions = get_bboxes(gt_frame_path)
                
                for idx,data in enumerate(annotatation_frame):

                    x1,y1,x2,y2,x3,y3,x4,y4 = data
                    content = contents[idx]

                    caption = is_captions[idx]
                    id_content = str(content)
                    if caption == "背景文字":
                        points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)
                        cv2.polylines(frame, [points], True, (0, 0, 255), thickness=3)
#                         frame=cv2AddChineseText(frame,id_content, (int(x1), int(y1) - 20),(0, 255, 0), 35)
                    else:
                        points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)
                        cv2.polylines(frame, [points], True, (0, 255, 0), thickness=3)
#                         frame=cv2AddChineseText(frame,id_content, (int(x1), int(y1) - 20),(0, 255, 0), 35)
                        
                        
#                     cv2.putText(frame, id_content, (int(x1), int(y1) - 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.5,
#                                 color=(0, 255, 0), thickness=2)

                frame_vis_path = os.path.join(output_vis1, frame_name.replace(".txt",".jpg"))
                cv2.imwrite(frame_vis_path, frame)
#             break
#         break
            
#         Frames2Video(frames_dir=video_vis_path)
        
        
        
        