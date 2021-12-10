# -*- coding: utf-8 -*-
import cv2
import os
import copy
import numpy as np
import math
from cv2 import VideoWriter, VideoWriter_fourcc
from tqdm import tqdm
from collections import OrderedDict
from shapely.geometry import Polygon, MultiPoint
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

def get_bboxes( data_polygt):
    bboxes = []
    text_tags = []
    with open(data_polygt, encoding='utf-8', mode='r') as f:
        for line in f.readlines():
            params = line.strip().strip('\ufeff').strip('\xef\xbb\xbf').split(':')
            try:
                label = params[8]
                if label == '*' or label == '###':
                    text_tags.append(True)
                else:
                    text_tags.append(False)
                # if label == '*' or label == '###':
                box = list(map(float, params[:8]))

                bboxes.append(box)

            except:
                print('load label failed on {}'.format(line))
    return bboxes, text_tags

# 普通 dict 插入元素时是无序的，使用 OrderedDict 按元素插入顺序排序
# 对字典按key排序, 默认升序, 返回 OrderedDict
def sort_key(old_dict, reverse=False):
    """对字典按key排序, 默认升序, 不修改原先字典"""
    # 先获得排序后的key列表
    keys = [int(i) for i in old_dict.keys()]
    keys = sorted(keys, reverse=reverse)
    # 创建一个新的空字典
    new_dict = OrderedDict()
    # 遍历 key 列表
    for key in keys:
        new_dict[str(key)] = old_dict[str(key)]
    return new_dict

def calculate_iou_polygen(bbox1, bbox2):
    '''
    :param bbox1: [x1, y1, x2, y2, x3, y3, x4, y4]
    :param bbox2:[x1, y1, x2, y2, x3, y3, x4, y4]
    :return:
    '''
    bbox1 = np.array([bbox1[0], bbox1[1],bbox1[2], bbox1[3],bbox1[4], bbox1[5],bbox1[6], bbox1[7]]).reshape(4, 2)
    poly1 = Polygon(bbox1).convex_hull  # python四边形对象，会自动计算四个点，最后四个点顺序为：左上 左下 右下 右上 左上
    bbox2 = np.array([bbox2[0], bbox2[1],bbox2[2], bbox2[3],bbox2[4], bbox2[5],bbox2[6], bbox2[7]]).reshape(4, 2)
    poly2 = Polygon(bbox2).convex_hull
    if poly1.area  < 0.01 or poly2.area < 0.01:
        return 0.0
    if not poly1.intersects(poly2):
        iou = 0
    else:
        inter_area = poly1.intersection(poly2).area
        union_area = poly1.area
        iou = float(inter_area) / union_area
    return iou

if __name__ == "__main__":
    
    orignal_annotation_path = "/share/caiyuanqiang/VideoSet"
    json_root = "/share/wuweijia/Data/MMVText/train/annotation"
    
    # ["Cls1_ZYDS","Cls2_ECY","Cls3_TY","Cls4_MRMX","Cls5_PP"]
    seqs = [
#             "Cls1_ZYDS","Cls2_ECY","Cls3_TY","Cls4_MRMX","Cls5_PP"
#         "Cls6_XW","Cls7_YX","Cls8_XJ","Cls9_XYH","Cls10_YL"
#         "Cls11_YS","Cls12_QG","Cls13_FCJJ","Cls14_CY","Cls15_SY"
#         "Cls16_ZWH","Cls17_JY","Cls18_LX","Cls19_SS","Cls20_XY"
#         "Cls21_MY","Cls22_QC","Cls23_HW","Cls24_YY","Cls25_DJ"
        "Cls26_KJ","Cls27_KP","Cls28_MY","Cls29_MZ","Cls30_WD"
    ]
    
    for cls in tqdm(os.listdir(orignal_annotation_path)):
        if cls.replace("_GtTxtsR4Frames","") not in seqs:
            continue
            
        if cls.split("_")[-1] == "GtTxtsR4Frames":
            video_path = os.path.join(orignal_annotation_path,cls)
            
            output_cls = os.path.join(json_root,cls)
            if not os.path.exists(output_cls):
                os.makedirs(output_cls)
                
            for video_name in os.listdir(video_path):
                annotation = {}
                each_video = os.path.join(video_path,video_name)
                
                for txt_name in os.listdir(each_video):
                    text_path = os.path.join(each_video,txt_name)
                    
                    boxes = []
                    filter_ =  []
                    try:
                        with open(text_path, encoding='utf-8', mode='r') as f:
                            for line in f.readlines():
                                object_boxes = []
                                params = line.strip().strip('\ufeff').strip('\xef\xbb\xbf').split(':')
#                                 try:
                                for x in params[:8]:
                                    object_boxes.append(x)
                                object_boxes.append(params[8])
                                if params[9] == "前景文字":
                                    object_boxes.append("caption")
                                else:
                                    object_boxes.append("scene")


                                object_boxes.append(params[10])

                                flag=1
                                for idx,box in enumerate(filter_):
                                    iou = calculate_iou_polygen(box,params[:8])
                                    if iou > 1.1:
                                        filter_[idx] = params[:8]
                                        flag = 0
                                        break
                                    elif iou > 0.18:
                                        flag = 0
                                        break
                                if flag == 1:

                                    filter_.append(params[:8])
                                    boxes.append(object_boxes)
  
                    except:
                        print(text_path)
                        assert False
                    frame_id = int(txt_name.split("_")[-1].split(".")[0])
                    annotation.update({str(frame_id):boxes})               
#                 annotation = sorted(annotation)
                annotation = sort_key(annotation)
                Generate_Json_annotation(annotation,os.path.join(output_cls,video_name+".json"))
   
    
    
    
    