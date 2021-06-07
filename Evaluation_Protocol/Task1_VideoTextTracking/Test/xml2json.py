# -*- coding: utf-8 -*-
import cv2
import os
import copy
import numpy as np
import math
import Levenshtein
from cv2 import VideoWriter, VideoWriter_fourcc
try:
    import xml.etree.cElementTree as ET  #解析xml的c语言版的模块
except ImportError:
    import xml.etree.ElementTree as ET
from tqdm import tqdm


class StorageDictionary(object):
    @staticmethod
    def dict2file(file_name, data_dict):
        try:
            import cPickle as pickle
        except ImportError:
            import pickle
        # import pickle
        output = open(file_name,'wb')
        pickle.dump(data_dict,output)
        output.close()

    @staticmethod
    def file2dict(file_name):
        try:
            import cPickle as pickle
        except ImportError:
            import pickle
        # import pickle
        pkl_file = open(file_name, 'rb')
        data_dict = pickle.load(pkl_file)
        pkl_file.close()
        return data_dict

    #Python语言特定的序列化模块是pickle，但如果要把序列化搞得更通用、更符合Web标准，就可以使用json模块
    @staticmethod
    def dict2file_json(file_name, data_dict):
        import json, io
        with io.open(file_name, 'w', encoding='utf-8') as fp:
            # fp.write(unicode(json.dumps(data_dict, ensure_ascii=False, indent=4) ) )  #可以解决在文件里显示中文的问题，不加的话是 '\uxxxx\uxxxx'
            fp.write((json.dumps(data_dict, ensure_ascii=False, indent=4) ) )


    @staticmethod
    def file2dict_json(file_name):
        import json, io
        with io.open(file_name, 'r', encoding='utf-8') as fp:
            data_dict = json.load(fp)
        return data_dict
    
def Generate_Json_annotation(TL_Cluster_Video_dict, Outpu_dir):
    '''   '''
    ICDAR21_DetectionTracks = {}
    text_id = 1
    for frame in TL_Cluster_Video_dict.keys():
        ICDAR21_DetectionTracks[frame] = []
        
        for text_list in TL_Cluster_Video_dict[frame]:
                
            ICDAR21_DetectionTracks[frame].append(text_list)

    StorageDictionary.dict2file_json(Outpu_dir, ICDAR21_DetectionTracks)
    
if __name__ == "__main__":
    
    orignal_annotation_path = "/share/ICDAR2021/VideoSpotting/SVTS/val"
    json_path = "./json_annotation"
    
    
    train_gt_dir = os.path.join(orignal_annotation_path,"annotations")
    train_data_dir = os.path.join(orignal_annotation_path,"images")  
    print("Data preparing...")
    for video_name in os.listdir(train_data_dir):
        annotation = {}
        
        video_path = os.path.join(train_data_dir,video_name)
        annotation_path = os.path.join(train_gt_dir,video_name+".xml")
        utf8_parser = ET.XMLParser(encoding='gbk')
        with open(annotation_path,'r',encoding='gbk') as load_f:
            tree = ET.parse(load_f, parser=utf8_parser)
        root = tree.getroot()#获取树型结构的根
        for child in root:
            # child.attrib["ID"]： 每一帧的序号
            image_path = os.path.join(video_path,child.attrib["ID"]+".jpg")
            boxes = []
            for children in child:
                # children.attrib： 文本实例的 ID号 ， Language ， Quality ， Transcription  ID
                object_boxes = []
                points = []
                
                for point in children:
                    object_boxes.append(point.attrib["x"])
                    object_boxes.append(point.attrib["y"])
                object_boxes.append(children.attrib["ID"])
                object_boxes.append(children.attrib["Transcription"])
                boxes.append(object_boxes)

            annotation.update({child.attrib["ID"]:boxes})
         
        
        Generate_Json_annotation(annotation,os.path.join(json_path,video_name+".json"))
    
    
    
    