#-*-coding:utf-8
import os
import math
import numpy as np
import cv2
import copy
import Levenshtein
from PIL import Image,ImageDraw,ImageFont
from shapely.geometry import Polygon, MultiPoint

'''
步骤一：采样视频帧标注信息平滑
步骤二：采样标注恢复为逐帧标注
步骤三：标注及补充标注信息与算法检测识别结果的相互修正
'''


class SortPoint(object):
    '''
    主要是对点的顺序进行修正的一些功能性函数
    '''
    @staticmethod
    def revise_point_seq_by_area(poly):
        '''
        通过面积公式对点的顺序进行修正
        :param poly: numpy.array( [ [x1,y1], [x2,y2], [x3,y3], [x4,y4] ] )
        :return: numpy.array( [ [x1,y1], [x2,y2], [x3,y3], [x4,y4] ] )
        '''
        # 0,1,2,3
        poly = np.array(poly)

        poly1 = poly[(0, 1, 3, 2), :]
        poly2 = poly[(0, 2, 3, 1), :]
        poly3 = poly[(0, 2, 1, 3), :]
        poly4 = poly[(0, 3, 1, 2), :]
        poly5 = poly[(0, 3, 2, 1), :]

        p_area = abs(SortPoint.polygon_area(poly))
        p_area_1 = abs(SortPoint.polygon_area(poly1))
        p_area_2 = abs(SortPoint.polygon_area(poly2))
        p_area_3 = abs(SortPoint.polygon_area(poly3))
        p_area_4 = abs(SortPoint.polygon_area(poly4))
        p_area_5 = abs(SortPoint.polygon_area(poly5))

        poly_list = []
        poly_list.append(poly)
        poly_list.append(poly1)
        poly_list.append(poly2)
        poly_list.append(poly3)
        poly_list.append(poly4)
        poly_list.append(poly5)

        idx_max = np.argmax([p_area, p_area_1, p_area_2, p_area_3, p_area_4, p_area_5])

        box_poly = poly_list[idx_max]
        dis_list = [x + y for [x, y] in box_poly]
        area_idx = np.argmin(dis_list)
        final_poly = box_poly[(area_idx, (area_idx + 1) % 4, (area_idx + 2) % 4, (area_idx + 3) % 4), :]

        return final_poly

    @staticmethod
    def polygon_area(poly):
        '''
        compute area of a polygon
        :param poly:
        :return:
        '''
        edge = [
            (poly[1][0] - poly[0][0]) * (poly[1][1] + poly[0][1]),
            (poly[2][0] - poly[1][0]) * (poly[2][1] + poly[1][1]),
            (poly[3][0] - poly[2][0]) * (poly[3][1] + poly[2][1]),
            (poly[0][0] - poly[3][0]) * (poly[0][1] + poly[3][1])
        ]
        return np.sum(edge) / 2.

    @staticmethod
    def check_and_validate_polys(poly):
        '''
        check so that the text poly is in the same direction,
        and also filter some invalid polygons
        :param polys: only one
        :param tags:
        :return:
        '''
        # (h, w) = (720, 1280)
        if poly.shape[0] == 0:
            return poly
        # poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)
        # poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)

        validated_polys = []
        # validated_tags = []
        # for poly, tag in zip(polys, tags):
        p_area = SortPoint.polygon_area(poly)
        if abs(p_area) < 1:
            # print poly
            print('invalid poly')
            return []
            # continue
        if p_area > 0:
            print('poly in wrong direction')
            poly = poly[(0, 3, 2, 1), :]
        # validated_polys.append(poly)
        # validated_tags.append(tag)
        return poly  # numpy.array(validated_polys)#, numpy.array(validated_tags)

    @staticmethod
    def revise_point_seq(poly):
        '''
        对点的顺序进行修正
        采用的方法是：DMP论文里面的 Sequential protocol of coordinates.
        :param poly: numpy.array( [ [x1,y1], [x2,y2], [x3,y3], [x4,y4] ] )
        :return: numpy.array( [ [x1,y1], [x2,y2], [x3,y3], [x4,y4] ] )
        方法步骤：
        1、找到第一个点；
        2、连线，找到第三个点，有中间的倾斜度的线；
        3、计算直线，将剩下的点代入直线方程，大于0的为第二个点，小于0的为第四个点；
        '''
        del_poly = poly.copy()
        p_capacity = np.array([x + y for [x, y] in poly])
        idx_first = np.argmin(p_capacity)
        [nx1, ny1] = poly[idx_first]
        del del_poly[idx_first]
        [[tx2, ty2], [tx3, ty3], [tx4, ty4]] = del_poly
        z2 = np.polyfit([nx1, tx2], [ny1, ty2], 1)
        z3 = np.polyfit([nx1, tx3], [ny1, ty3], 1)
        z4 = np.polyfit([nx1, tx4], [ny1, ty4], 1)

        p2 = np.poly1d(z2)
        p3 = np.poly1d(z3)
        p4 = np.poly1d(z4)

        k_list = np.array([z2[0], z3[0], z4[0]])
        p_list = [p2, p3, p4]
        sort_k_list = np.argsort(k_list)
        idx_third = sort_k_list[1]
        idx_second = sort_k_list[0]
        [nx2, ny2] = del_poly[idx_second]
        [nx3, ny3] = del_poly[idx_third]
        [nx4, ny4] = del_poly[sort_k_list[-1]]

        print(np.array([[nx1, ny1], [nx2, ny2], [nx3, ny3], [nx4, ny4]]))
        return np.array([[nx1, ny1], [nx2, ny2], [nx3, ny3], [nx4, ny4]])

def change_cv2_draw(image,strs,local,sizes=24,colour=(0,255,0)):
    cv2img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pilimg = Image.fromarray(cv2img)
    draw = ImageDraw.Draw(pilimg)  # 图片上打印
    font = ImageFont.truetype("./HuaWenKaiTi.ttf",sizes, encoding="utf-8")
    draw.text(local, strs, colour, font=font)
    image = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
    return image

def get_file_path_list(json_dir, postfix = [".jpg"] ):
    '''  '''
    file_path_list = []
    if os.path.exists(json_dir):
        print (json_dir)
    else:
        print ("no exist")
    for rt, dirs, files in os.walk(json_dir):
        if len(dirs)>0:
            continue
        for file in files:
            if os.path.splitext(file)[1] in postfix:
                file_path_list.append( os.path.join(rt, file ) )
    return file_path_list

def write_4points(file, data):
    ''' x1,y1,x2,y2,x3,y3,x4,y4, extraContent, option'''
    with open(file, 'w', encoding='utf-8') as fw:
        for line in data:
            line = [str(e) for e in line]
            wline = ":".join(line)
            wline += "\n"
            fw.write(wline)

def load_4points2items(file):
    '''  x1,y1,x2,y2,x3,y3,x4,y4, extraContent, option '''
    data_list = []
    with open(file, 'r', encoding='utf-8') as fr:
        for line in fr.readlines():
            line = line.strip('\n')
            data = line.split(':')
            if len(data)!=10 or data[8] == "#1": # or data[8] == '#nuII':
                continue
            data_list.append(data)
    return data_list

def calculate_iou(bbox1, bbox2):
    '''
    :param bbox1: [x1, y1, x2, y2, x3, y3, x4, y4]
    :param bbox2:[x1, y1, x2, y2, x3, y3, x4, y4]
    :return:
    '''
    bbox1 = np.array([bbox1[0], bbox1[1],
                      bbox1[6], bbox1[7],
                      bbox1[4], bbox1[5],
                      bbox1[2], bbox1[3]]).reshape(4, 2)
    poly1 = Polygon(bbox1).convex_hull  # python四边形对象，会自动计算四个点，最后四个点顺序为：左上 左下 右下 右上 左上
    bbox2 = np.array([bbox2[0], bbox2[1],
                      bbox2[6], bbox2[7],
                      bbox2[4], bbox2[5],
                      bbox2[2], bbox2[3]]).reshape(4, 2)
    poly2 = Polygon(bbox2).convex_hull
    if poly1.area  < 0.01 or poly2.area < 0.01:
        return 0.0
    if not poly1.intersects(poly2):
        iou = 0
    else:
        inter_area = poly1.intersection(poly2).area
        union_area = poly1.area + poly2.area - inter_area
        iou = float(inter_area) / union_area
    return iou

def IsInCluster(frame_id, obj_id, TL_Cluster_dict, data, TIoU=0.3, FrameGap=2, TSimEdit=0.3):
    '''  判断新文字目标是否可以合并到已有类簇中
    :param TL_Cluster_dict:
    :param data:
    :return:
    '''

    if len(TL_Cluster_dict) == 1:
        return False

    new_data = copy.deepcopy(data)
    new_data.append(obj_id)
    new_data.append(frame_id)

    current_BBox = data[:8]
    current_Content = data[8]
    current_Cls = data[9]
    max_iou = 0.0
    max_iou_sim = 0.0
    max_sim = 0.0
    max_sim_iou = 0.0
    max_cluster_key = 0
    max_cluster_strkey = 0

    for cluster_key, cluster_dict in TL_Cluster_dict.items():
        if cluster_key == "cluster_num":
            continue

        center_data = cluster_dict["cluster_center"]
        end_frame_id = cluster_dict["end_frame_id"]
        if not 5< (frame_id - int(end_frame_id)) < 15:
            continue

        center_BBox = center_data[:8]
        center_Content = center_data[8]
        center_Cls = center_data[9]

        curIoU = calculate_iou(center_BBox, current_BBox)
        curEdit = Levenshtein.distance(str(center_Content), str(current_Content))
        # print(center_Content, current_Content, curEdit, len(center_Content),len(current_Content))
        SimContent = 1.0-((curEdit*2)/(len(center_Content)+len(current_Content)))
        if curIoU > max_iou:
            max_iou = curIoU
            max_sim_iou = SimContent
            max_cluster_key = cluster_key
        if SimContent>max_sim:
            max_sim = SimContent
            max_iou_sim = curIoU
            max_cluster_strkey = cluster_key

    if current_Cls == '背景文字':
        # print('背景文字',current_Cls, current_Content, max_iou)
        if max_iou > TIoU:
            TL_Cluster_dict[max_cluster_key]["cluster_center"] = new_data
            TL_Cluster_dict[max_cluster_key]["end_frame_id"] = frame_id
            TL_Cluster_dict[max_cluster_key]["element_list"].append(new_data)
            return True
        elif max_sim>TSimEdit and max_iou_sim>0.005:
            TL_Cluster_dict[max_cluster_strkey]["cluster_center"] = new_data
            TL_Cluster_dict[max_cluster_strkey]["end_frame_id"] = frame_id
            TL_Cluster_dict[max_cluster_strkey]["element_list"].append(new_data)
            return True
        else:
            return False
    else:
        # print('前景文字', current_Cls, current_Content, max_iou, max_sim_iou)
        if max_iou > 0.5 and max_sim_iou>0.5:
            TL_Cluster_dict[max_cluster_key]["cluster_center"] = new_data
            TL_Cluster_dict[max_cluster_key]["end_frame_id"] = frame_id
            TL_Cluster_dict[max_cluster_key]["element_list"].append(new_data)
            return True
        elif max_sim>0.98 and max_iou_sim>0.005:
            TL_Cluster_dict[max_cluster_strkey]["cluster_center"] = new_data
            TL_Cluster_dict[max_cluster_strkey]["end_frame_id"] = frame_id
            TL_Cluster_dict[max_cluster_strkey]["element_list"].append(new_data)
            return True
        else:
            return False

def CreateNewCluster(frame_id, obj_id, TL_Cluster_dict, data):
    '''
    Create a new cluster
    TL_Cluster_dict = {}
    {
        "cluster_num": int
        "cluster_id": {...}
        "cluster_id": {...}
    }
    "cluster_id":
    {
     "start_frame_id": int
     "end_frame_id" : int
    "cluster_center": [data, obj_id,frame_id]
    "element_list": [[data, obj_id,frame_id], [...], ...]
    ""
    }
    :param frame_id:
    :param obj_id:
    :param TL_Cluster_dict:
    :param data:
    :return:
    '''
    # Add the id of object and frame to the new text instance.
    new_data = copy.deepcopy(data)
    new_data.append(obj_id)
    new_data.append(frame_id)

    # Apply the new ID for new cluster.
    new_cluster_id = TL_Cluster_dict["cluster_num"] + 1

    # Create a new cluster
    cluster_dict = {}
    cluster_dict["cluster_center"] = new_data
    cluster_dict["start_frame_id"] = frame_id
    cluster_dict["end_frame_id"] = frame_id
    cluster_dict["element_list"] = []
    cluster_dict["element_list"].append(new_data)

    # Add the new cluster to the dictionary of cluster, and update the number of cluster.
    TL_Cluster_dict[new_cluster_id] = copy.deepcopy(cluster_dict)
    TL_Cluster_dict["cluster_num"] += 1

def InsertElement(data1, data2, temp_obj_id):
    '''  '''
    # data1 = [float(e) for e in data1]
    # data2 = [float(e) for e in data2]
    frame_idx = int(data1[-1])
    fframe_idx = int(data2[-1])

    data = []
    for e1, e2 in zip(data1[:8], data2[:8]):
        data.append((float(e1) + float(e2)) / 2)

    data[:8] = [int(e) for e in data[:8]]
    data.append(data1[8])

    data_list = []
    temp_frame_idx = frame_idx
    for i in range(fframe_idx - frame_idx - 1):
        temp_data = copy.deepcopy(data[:9])
        temp_obj_id += 1
        temp_frame_idx += 1
        temp_data.append(int(temp_obj_id))
        temp_data.append(int(temp_frame_idx))
        data_list.append(copy.deepcopy(temp_data))
        temp_data = []
        assert temp_frame_idx < fframe_idx
    return data_list

def RevisePoints(data_list):
    '''
    data = x1,y1,x2,y2,x3,y3,x4,y4, extraContent, option, obj_id, frame_id
    [[data, obj_id,frame_id], [...], ...]
    0、修正点的顺序，左上角为起始点，然后顺时针排序.
    '''
    re_data_list = []
    for idx, data in enumerate(data_list):
        x1, y1, x2, y2, x3, y3, x4, y4, extraContent, option, obj_id, frame_id = data
        if extraContent == "#1" or extraContent == '#nuII':
            continue
        poly = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)
        revised_poly = SortPoint.revise_point_seq_by_area(poly)
        reorder_revised_poly = SortPoint.check_and_validate_polys(revised_poly)
        if len(reorder_revised_poly) == 0:
            continue
        [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] = reorder_revised_poly
        x1, y1, x2, y2, x3, y3, x4, y4 = [str(e) for e in [x1, y1, x2, y2, x3, y3, x4, y4] ]
        re_data_list.append( [x1, y1, x2, y2, x3, y3, x4, y4, extraContent, option, obj_id, frame_id])

    return re_data_list

def ReviseClass(data_list):
    '''
    data = x1,y1,x2,y2,x3,y3,x4,y4, extraContent, option, obj_id, frame_id
    [[data, obj_id,frame_id], [...], ...]
    1、类别信息修正；
    '''
    if 1 >= len(data_list):
        return data_list
    if 2 == len(data_list):
        data1 = data_list[0]
        data2 = data_list[1]
        if data1[9] != data2[9]: #前景文字不可能一闪而过
            for idx, data in enumerate(data_list):
                data_list[idx][9] = '背景文字'
    cls_dict = {}
    cls_dict['前景文字'] = 0
    cls_dict['背景文字'] = 0
    for data in data_list:
        if data[9] != '前景文字' or data[9] != '背景文字':
            continue
        cls_dict[data[9]] += 1
    if cls_dict['前景文字'] > cls_dict['背景文字']:
        for idx, data in enumerate(data_list):
            data_list[idx][9] = '前景文字'
    else:
        for idx, data in enumerate(data_list):
            data_list[idx][9] = '背景文字'

    return data_list

def ReviseContent(data_list):
    '''
    data = x1,y1,x2,y2,x3,y3,x4,y4, extraContent, option, obj_id, frame_id
    [[data, obj_id,frame_id], [...], ...]
    2、识别内容信息修正；
    '''
    if 1 >= len(data_list):
        return data_list
    for idx in range(len(data_list)-1):
        if len(data_list[idx][8]) < len(data_list[idx+1][8]):  # 短的更有可能全对
            data_list[idx+1][8] = data_list[idx][8]
        else:
            data_list[idx][8] = data_list[idx+1][8]

    # cls_dict = {}
    # for data in data_list:
    #     if data[8] not in cls_dict.keys():
    #         cls_dict[data[8]] = 0
    #     cls_dict[data[8]] += 1
    # # 排序找出出现次数最多的识别内容.
    # print(cls_dict)
    # content = sorted(cls_dict.items(), key=lambda item:item[1], reverse=True)[0][0]
    # for idx, data in enumerate(data_list):
    #     data_list[idx][8] = content

    return data_list

def ReviseBBox(data_list):
    '''
    data = x1,y1,x2,y2,x3,y3,x4,y4, extraContent, option, obj_id, frame_id
    [[data, obj_id,frame_id], [...], ...]
    3、位置信息修正；
    '''
    if 3 > len(data_list):
        return data_list
    if data_list[0][9] == '前景文字':
        return data_list

    RBox = []
    for idx, data in enumerate(data_list):
        x1, y1, x2, y2, x3, y3, x4, y4, extraContent, option, obj_id, frame_id = data
        if x1==x4 and x2==x3 and y1==y2 and y3==y4:
            RBox.append((idx,0))   # 矩形
        else:
            RBox.append((idx,1))   # 四边形

    for (idx,rr) in RBox:
        if rr == 0:
            pass

    return data_list

#对同一个文字目标进行平滑处理
def SmoothObjects(TL_Cluster_dict):
    '''  对同一个文字目标进行平滑处理:
         0、修正点的顺序
         1、类别信息修正；
         2、识别内容信息修正；
         3、位置信息修正；
    '''
    # print(len(TL_Cluster_dict))
    for cluster_id, cluster_dict in TL_Cluster_dict.items():
        if cluster_id == 'cluster_num':
            continue
        # print(cluster_id,len(cluster_dict),len(cluster_dict['element_list']),'=======\n')
        # print(cluster_dict)
        data_list = cluster_dict['element_list']
        data_list = RevisePoints(data_list)
#         data_list = ReviseClass(data_list)
        data_list = ReviseContent(data_list)
        data_list = ReviseBBox(data_list)
        TL_Cluster_dict[cluster_id]['element_list'] = copy.deepcopy(data_list)

        # for data in data_list:
        #     print(data)

def InsertPair(data1, data2):
    '''
    data = x1,y1,x2,y2,x3,y3,x4,y4, extraContent, option, obj_id, frame_id
    '''
    insert_list = []
    x1_1, y1_1, x2_1, y2_1, x3_1, y3_1, x4_1, y4_1, extraContent_1, option_1, obj_id_1, frame_id_1 = data1
    x1_2, y1_2, x2_2, y2_2, x3_2, y3_2, x4_2, y4_2, extraContent_2, option_2, obj_id_2, frame_id_2 = data2
    das1 = x1_1, y1_1, x2_1, y2_1, x3_1, y3_1, x4_1, y4_1, obj_id_1, frame_id_1
    das2 = x1_2, y1_2, x2_2, y2_2, x3_2, y3_2, x4_2, y4_2, obj_id_2, frame_id_2
    x1_1, y1_1, x2_1, y2_1, x3_1, y3_1, x4_1, y4_1, obj_id_1, frame_id_1 = [int(e) for e in das1]
    x1_2, y1_2, x2_2, y2_2, x3_2, y3_2, x4_2, y4_2, obj_id_2, frame_id_2 = [int(e) for e in das2]

    num = abs(frame_id_2 - frame_id_1)
    x1_s = (x1_2 - x1_1)/num
    y1_s = (y1_2 - y1_1)/num
    x2_s = (x2_2 - x2_1)/num
    y2_s = (y2_2 - y2_1)/num
    x3_s = (x3_2 - x3_1)/num
    y3_s = (y3_2 - y3_1)/num
    x4_s = (x4_2 - x4_1)/num
    y4_s = (y4_2 - y4_1)/num
    for idx in range(1,num):
        x1_n = x1_1 + x1_s*idx
        y1_n = y1_1 + y1_s*idx
        x2_n = x2_1 + x2_s*idx
        y2_n = y2_1 + y2_s*idx
        x3_n = x3_1 + x3_s*idx
        y3_n = y3_1 + y3_s*idx
        x4_n = x4_1 + x4_s*idx
        y4_n = y4_1 + y4_s*idx
        extraContent_n = extraContent_1
        option_n = option_1
        obj_id_n = obj_id_1
        frame_id_n = frame_id_1+idx
        x1_n, y1_n, x2_n, y2_n, x3_n, y3_n, x4_n, y4_n = [round(e) for e in [x1_n, y1_n, x2_n, y2_n, x3_n, y3_n, x4_n, y4_n]]
        data_n = [x1_n, y1_n, x2_n, y2_n, x3_n, y3_n, x4_n, y4_n, extraContent_n, option_n, obj_id_n, frame_id_n]
        data_n[:-2] = [str(e) for e in data_n[:-2]]
        insert_list.append(data_n)

    return insert_list, num

def calculate_offset(data_1, data_2):
    '''
    data = x1,y1,x2,y2,x3,y3,x4,y4, extraContent, option, obj_id, frame_id
    '''
    insert_list = []
    x1_1, y1_1, x2_1, y2_1, x3_1, y3_1, x4_1, y4_1, extraContent_1, option_1, obj_id_1, frame_id_1 = data_1
    x1_2, y1_2, x2_2, y2_2, x3_2, y3_2, x4_2, y4_2, extraContent_2, option_2, obj_id_2, frame_id_2 = data_2
    das1 = x1_1, y1_1, x2_1, y2_1, x3_1, y3_1, x4_1, y4_1, obj_id_1, frame_id_1
    das2 = x1_2, y1_2, x2_2, y2_2, x3_2, y3_2, x4_2, y4_2, obj_id_2, frame_id_2
    x1_1, y1_1, x2_1, y2_1, x3_1, y3_1, x4_1, y4_1, obj_id_1, frame_id_1 = [int(e) for e in das1]
    x1_2, y1_2, x2_2, y2_2, x3_2, y3_2, x4_2, y4_2, obj_id_2, frame_id_2 = [int(e) for e in das2]

    num = abs(frame_id_2 - frame_id_1)
    x1_s = (x1_2 - x1_1)/num
    y1_s = (y1_2 - y1_1)/num
    x2_s = (x2_2 - x2_1)/num
    y2_s = (y2_2 - y2_1)/num
    x3_s = (x3_2 - x3_1)/num
    y3_s = (y3_2 - y3_1)/num
    x4_s = (x4_2 - x4_1)/num
    y4_s = (y4_2 - y4_1)/num

    offset_list = [x1_s, y1_s, x2_s, y2_s, x3_s, y3_s, x4_s, y4_s]

    return offset_list

def extract_crop(box, frame):
    '''
    在frame 中截切 box 区域
    box = x1,y1,x2,y2,x3,y3,x4,y4
    '''
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    x1, y1, x2, y2, x3, y3, x4, y4 = [int(e) for e in box]
    min_x = min([x1,x2,x3,x3])
    max_x = max([x1,x2,x3,x3])
    min_y = min([y1,y2,y3,y4])
    max_y = max([y1,y2,y3,y4])
    crop = frame[min_y:max_y, min_x:max_x]

    return crop

def calculate_prebox(box, offset):
    '''
    box = x1,y1,x2,y2,x3,y3,x4,y4
    offset = x1_s, y1_s, x2_s, y2_s, x3_s, y3_s, x4_s, y4_s
    '''
    x1, y1, x2, y2, x3, y3, x4, y4 = [int(e) for e in box]
    x1_s, y1_s, x2_s, y2_s, x3_s, y3_s, x4_s, y4_s = offset

    x1_n = x1 - x1_s
    y1_n = y1 - y1_s
    x2_n = x2 - x2_s
    y2_n = y2 - y2_s
    x3_n = x3 - x3_s
    y3_n = y3 - y3_s
    x4_n = x4 - x4_s
    y4_n = y4 - y4_s

    x1_n = max(x1_n, 1)
    x2_n = max(x2_n, 1)
    x3_n = max(x3_n, 1)
    x4_n = max(x4_n, 1)

    y1_n = max(y1_n, 1)
    y2_n = max(y2_n, 1)
    y3_n = max(y3_n, 1)
    y4_n = max(y4_n, 1)

    return [x1_n, y1_n, x2_n, y2_n, x3_n, y3_n, x4_n, y4_n]

def calculate_rearbox(box, offset):
    '''
    box = x1,y1,x2,y2,x3,y3,x4,y4
    offset = x1_s, y1_s, x2_s, y2_s, x3_s, y3_s, x4_s, y4_s
    '''
    x1, y1, x2, y2, x3, y3, x4, y4 = [int(e) for e in box]
    x1_s, y1_s, x2_s, y2_s, x3_s, y3_s, x4_s, y4_s = offset

    x1_n = x1 + x1_s
    y1_n = y1 + y1_s
    x2_n = x2 + x2_s
    y2_n = y2 + y2_s
    x3_n = x3 + x3_s
    y3_n = y3 + y3_s
    x4_n = x4 + x4_s
    y4_n = y4 + y4_s

    x1_n = max(x1_n, 1)
    x2_n = max(x2_n, 1)
    x3_n = max(x3_n, 1)
    x4_n = max(x4_n, 1)

    y1_n = max(y1_n, 1)
    y2_n = max(y2_n, 1)
    y3_n = max(y3_n, 1)
    y4_n = max(y4_n, 1)

    return [x1_n, y1_n, x2_n, y2_n, x3_n, y3_n, x4_n, y4_n]

def calculate_L2Dis(img1, img2):
    '''
    计算两个图片之间的L2距离.
    '''
    if type(img2) == 'NoneType':
        return 100.0
    h, w = img1.shape[0:2]
    if h<8 or w < 8:
        return 100.0
    img2 = cv2.resize(img2,(w,h))
    img1 = np.array(img1)
    img2 = np.array(img2)
    MeanL2 = np.sum(np.square(img1 - img2)) / (h*w)

    return MeanL2

def ComplementMedium(data_list):
    '''
    1、补充采样点之间的文字；
    data = x1,y1,x2,y2,x3,y3,x4,y4, extraContent, option, obj_id, frame_id
    '''
    if 1 > len(data_list):
        return data_list
    re_data_list = copy.deepcopy(data_list)
    for idx in range(0,len(data_list)-1):
        insert_list, num = InsertPair(data_list[idx],data_list[idx+1])
        for jdx in range(0,num-1):
            ndx = num*idx + (jdx+1)
            re_data_list.insert(ndx,insert_list[jdx])

    return re_data_list

def ComplementStartEnd(data_list, FrameId_FramePath_dict, TL2=50):
    '''
    2 and 3、寻找起始点 和 结束点；
    data = x1,y1,x2,y2,x3,y3,x4,y4, extraContent, option, obj_id, frame_id
    '''
    re_data_list = copy.deepcopy(data_list)
    num_frame = len(FrameId_FramePath_dict)

    if 1 > len(data_list):
        return data_list
    elif 1 == len(data_list):
        # fixme: 寻找起始点
        data_start = data_list[0]

        comp_num = 0
        x1, y1, x2, y2, x3, y3, x4, y4, extraContent, option, obj_id, frame_id = data_start
        current_frame_id = frame_id
        current_box = [x1, y1, x2, y2, x3, y3, x4, y4]
        current_frame = cv2.imread(FrameId_FramePath_dict[current_frame_id])
        current_crop = extract_crop(current_box, current_frame)

        while comp_num < 7:
            if current_frame_id == 1:
                break
            previous_frame_id = current_frame_id - 1
            previous_frame = cv2.imread(FrameId_FramePath_dict[previous_frame_id])
            previous_box = current_box
            previous_crop = extract_crop(previous_box, previous_frame)

            # print(previous_frame_id, previous_box, previous_crop.shape)
            # cv2.imshow('previous_crop', previous_crop)
            # cv2.waitKey()

            try: 
                MeanL2 = calculate_L2Dis(current_crop, previous_crop)
            except:
                print(FrameId_FramePath_dict[previous_frame_id])
                MeanL2 = 0
                
            # print('MeanL2',MeanL2)
            if MeanL2 < TL2:
                current_frame_id = previous_frame_id
                current_box = previous_box
                current_crop = previous_crop
                x1_n, y1_n, x2_n, y2_n, x3_n, y3_n, x4_n, y4_n = [int(e) for e in previous_box]
                previous_text = [x1_n, y1_n, x2_n, y2_n, x3_n, y3_n, x4_n, y4_n, extraContent, option, obj_id,
                                 previous_frame_id]
                re_data_list.insert(0, previous_text)
            else:
                break
            comp_num += 1

        # fixme: 寻找结束点
        data_start = data_list[-1]

        comp_num = 0
        x1, y1, x2, y2, x3, y3, x4, y4, extraContent, option, obj_id, frame_id = data_start
        current_frame_id = frame_id
        current_box = [x1, y1, x2, y2, x3, y3, x4, y4]
        current_frame = cv2.imread(FrameId_FramePath_dict[current_frame_id])
        current_crop = extract_crop(current_box, current_frame)
        while comp_num < 7:
            if current_frame_id == num_frame:
                break
            rear_frame_id = current_frame_id + 1
            rear_frame = cv2.imread(FrameId_FramePath_dict[rear_frame_id])
            rear_box = current_box
            rear_crop = extract_crop(rear_box, rear_frame)
            
            try: 
                MeanL2 = calculate_L2Dis(current_crop, rear_crop)
            except:
                print(FrameId_FramePath_dict[rear_frame_id])
                MeanL2 = 0
            
            if MeanL2 < TL2:
                current_frame_id = rear_frame_id
                current_box = rear_box
                current_crop = rear_crop
                x1_n, y1_n, x2_n, y2_n, x3_n, y3_n, x4_n, y4_n = [int(e) for e in rear_box]
                rear_text = [x1_n, y1_n, x2_n, y2_n, x3_n, y3_n, x4_n, y4_n, extraContent, option, obj_id,
                             rear_frame_id]
                re_data_list.append(rear_text)
            else:
                break
            comp_num += 1
    else:
        #fixme: 寻找起始点
        data_start_1 = data_list[0]
        data_start_2 = data_list[1]
        offset_list = calculate_offset(data_start_1, data_start_2)

        sim_start = 0.0
        comp_num = 0
        x1, y1, x2, y2, x3, y3, x4, y4, extraContent, option, obj_id, frame_id = data_start_1
        current_frame_id = frame_id
        current_box = [x1, y1, x2, y2, x3, y3, x4, y4]
        current_frame = cv2.imread(FrameId_FramePath_dict[current_frame_id])
        current_crop = extract_crop(current_box, current_frame)

        # cv2.imshow('current_frame', current_frame)
        # cv2.waitKey()
        # print(current_box)
        # cv2.imshow('current_crop', current_crop)
        # cv2.waitKey()

        while comp_num < 7:
            if current_frame_id == 1:
                break
            previous_frame_id = current_frame_id-1
            previous_frame = cv2.imread(FrameId_FramePath_dict[previous_frame_id])
            previous_box = calculate_prebox(current_box, offset_list)
            previous_crop = extract_crop(previous_box, previous_frame)

            # print(previous_frame_id, previous_box, previous_crop.shape)
            # cv2.imshow('previous_crop', previous_crop)
            # cv2.waitKey()
            
            try: 
                MeanL2 = calculate_L2Dis(current_crop, previous_crop)
            except:
                print(FrameId_FramePath_dict[previous_frame_id])
                MeanL2 = 0
                
            
            # print('MeanL2',MeanL2)
            if MeanL2 < TL2:
                current_frame_id = previous_frame_id
                current_box = previous_box
                current_crop = previous_crop
                x1_n, y1_n, x2_n, y2_n, x3_n, y3_n, x4_n, y4_n = [int(e) for e in previous_box]
                previous_text = [x1_n, y1_n, x2_n, y2_n, x3_n, y3_n, x4_n, y4_n, extraContent, option, obj_id, previous_frame_id ]
                re_data_list.insert(0,previous_text)
            else:
                break
            comp_num += 1

        #fixme: 寻找结束点
        data_start_Last1 = data_list[-1]
        data_start_Last2 = data_list[-2]
        offset_list = calculate_offset(data_start_Last2, data_start_Last1)

        comp_num = 0
        x1, y1, x2, y2, x3, y3, x4, y4, extraContent, option, obj_id, frame_id = data_start_Last1
        current_frame_id = frame_id
        current_box = [x1, y1, x2, y2, x3, y3, x4, y4]
        current_frame = cv2.imread(FrameId_FramePath_dict[current_frame_id])
        current_crop = extract_crop(current_box, current_frame)
        while comp_num < 7:
            if current_frame_id == num_frame:
                break
            rear_frame_id = current_frame_id + 1
            rear_frame = cv2.imread(FrameId_FramePath_dict[rear_frame_id])
            rear_box = calculate_rearbox(current_box, offset_list)
            rear_crop = extract_crop(rear_box, rear_frame)
            
            try: 
                MeanL2 = calculate_L2Dis(current_crop, rear_crop)
            except:
                print(FrameId_FramePath_dict[rear_frame_id])
                MeanL2 = 0
                
            
            if MeanL2 < TL2:
                current_frame_id = rear_frame_id
                current_box = rear_box
                current_crop = rear_crop
                x1_n, y1_n, x2_n, y2_n, x3_n, y3_n, x4_n, y4_n = [int(e) for e in rear_box]
                rear_text = [x1_n, y1_n, x2_n, y2_n, x3_n, y3_n, x4_n, y4_n, extraContent, option, obj_id, rear_frame_id ]
                re_data_list.append(rear_text)
            else:
                break
            comp_num += 1

    return re_data_list

def ComplementSampleFrames(TL_Cluster_dict, VideoFrames_dir):
    '''
        对采样空白区域进行补充：
        1、补充采样点之间的文字；
        2、寻找起始点；
        3、寻找结束点。
    '''
    VideoFramesPath_list = get_file_path_list(VideoFrames_dir, ['.jpg'])
    FrameId_FramePath_dict = {}
    for frame_path in VideoFramesPath_list:
        frame_dir, frame_name = os.path.split(frame_path)
        name_prefix, name_postfix = os.path.splitext(frame_name)
        frame_id = int(name_prefix.split('_')[-1])
        FrameId_FramePath_dict[frame_id] = frame_path

    # print(len(TL_Cluster_dict))
    for cluster_id, cluster_dict in TL_Cluster_dict.items():
        if cluster_id == 'cluster_num':
            continue
        # print(cluster_id,len(cluster_dict),len(cluster_dict['element_list']),'=======\n')
        # print(cluster_dict)
        data_list = cluster_dict['element_list']
        data_list = ComplementMedium(data_list)
        data_list = ComplementStartEnd(data_list, FrameId_FramePath_dict)
        TL_Cluster_dict[cluster_id]['element_list'] = copy.deepcopy(data_list)

        # for data in data_list:
        #     print(data)

def Cluster2Frames(TL_Cluster_dict, Frames_dir):
    '''
    将类簇形式的文字目标 转换 为视频帧形式的文字目标
    并将GT写入txt文件中
    '''
    Cls_dir, Video_dir = os.path.split(Frames_dir)
    Cls_GtTxt_dir = Cls_dir.replace('_Frames','_GtTxtsR3Frames')
    if not os.path.exists(Cls_GtTxt_dir):
        os.mkdir(Cls_GtTxt_dir)

    VideoFramesPath_list = get_file_path_list(Frames_dir, ['.jpg'])
    NumFrames = len(VideoFramesPath_list)
    # print(len(TL_Cluster_dict))
    FramesGt_dict = {}
    for idx in range(1, NumFrames+1):
        FramesGt_dict[idx] = []

    for cluster_id, cluster_dict in TL_Cluster_dict.items():
        if cluster_id == 'cluster_num':
            continue
        for data in cluster_dict['element_list']:
            data[-2] = cluster_id     #同一个文字赋值同一个类簇id
            FramesGt_dict[data[-1]].append(data)

    for frame_idx, data_list in FramesGt_dict.items():
        GtTxt_name = Video_dir + "_%0*d"%(6,frame_idx) + ".txt"
        Cls_Video_GtTxt_dir = os.path.join(Cls_GtTxt_dir, Video_dir)
        if not os.path.exists(Cls_Video_GtTxt_dir):
            os.makedirs(Cls_Video_GtTxt_dir)
        GtTxt_path = os.path.join(Cls_Video_GtTxt_dir, GtTxt_name)
        write_4points(GtTxt_path, data_list)

    return FramesGt_dict

def VisualizeFrames(FramesGt_dict, Frames_dir):
    '''
    将插值内容可视化到视频帧上
    '''
    Cls_dir, Video_dir = os.path.split(Frames_dir)
    Cls_Vis_dir = Cls_dir.replace('_Frames','_VisualFrames')
    if not os.path.exists(Cls_Vis_dir):
        os.mkdir(Cls_Vis_dir)
    for frame_idx, data_list in FramesGt_dict.items():
        # if len(data_list) == 0:
        #     continue
        video_name = Video_dir + "_%0*d"%(6,frame_idx) + ".jpg"
        video_path = os.path.join(Frames_dir,video_name)
        frame = cv2.imread(video_path)

        for data in data_list:
            x1, y1, x2, y2, x3, y3, x4, y4, extraContent, option, obj_id, frame_id = data
            points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)
            cv2.polylines(frame, [points], True, (0, 0, 255), thickness=4)
            strs = str(extraContent) + "/" + str(option)
            # cv2.putText(frame, strs, (int(x1), int(y1) - 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.5,
            #             color=(0, 255, 0), thickness=2)
            frame = change_cv2_draw(frame, strs, (int(x4), int(y4)), sizes=55, colour=(255, 0, 0))
        video_vis_path = os.path.join(Cls_Vis_dir, Video_dir)
        if not os.path.exists(video_vis_path):
            os.makedirs(video_vis_path)
        frame_vis_path = os.path.join(video_vis_path, video_name)
        print(frame_vis_path)
        cv2.imwrite(frame_vis_path, frame)


def RecoveryVideoAnnotations(SampleAnno_dir, Frames_dir):
    '''
    处理单个视频：
    采样视频帧标注信息平滑：
    1、通过位置信息、识别内容、类别标签进行相同目标搜索；
    2、对每一个文字目标做平滑处理。
    采样标注恢复为逐帧标注：
    3、同一个文字目标在前后两个采样视频帧内同时存在；
    4、前采样视频帧内存在某一个文字目标，但是在后采样视频帧内不存在该文字目标；
    5、前采样视频帧内不存在某一个文字目标，但是在后采样视频帧内存在该文字目标。
    标注及补充标注信息与算法检测识别结果的相互修正：
    6、
    '''
    # SampleAnno_dir = "E:\\KSText\\KSText_VideoFrames_Results\\Cls5_PP_SampleFrames\\39734424643"
    # Frames_dir = "E:\\KSText\\KSText_VideoFrames_Results\\Cls5_PP_Frames\\39734424643"

    Sample_GtTxt_list = get_file_path_list(SampleAnno_dir, ['.txt'])
    if len(Sample_GtTxt_list) == 0:
        return
    TL_dict = {}
    TL_Cluster_dict = {}
    TL_Cluster_dict["cluster_num"] = 0
    for GtTxt_path in Sample_GtTxt_list:
        frame_name = os.path.split(GtTxt_path)[1]
        frame_id = int(os.path.splitext(frame_name)[0].split('_')[1])
        data_list = load_4points2items(GtTxt_path)
        TL_dict[frame_id] = {}
        TL_dict[frame_id]["txt_name"] = frame_name
        for obj_id, data in enumerate(data_list):
            TL_dict[frame_id][obj_id] = data
            isLink = IsInCluster(frame_id, obj_id, TL_Cluster_dict, data, TIoU=0.2)
            if isLink == True:
                # print("The cluster is updated, frame_id = {}, object_id = {}".format(frame_id, obj_id))
                pass
            else:  #
                CreateNewCluster(frame_id, obj_id, TL_Cluster_dict, data)
    #采样视频帧标注信息平滑
    SmoothObjects(TL_Cluster_dict)

    #采样标注恢复为逐帧标注
    ComplementSampleFrames(TL_Cluster_dict, Frames_dir)

    # 可视化
    FramesGt_dict = Cluster2Frames(TL_Cluster_dict, Frames_dir)
    # VisualizeFrames(FramesGt_dict, Frames_dir)

def BatchRecoveryVideoAnnotations():
    '''

    '''
    AllFrames_dir = "/share/caiyuanqiang/KSText_ReDetRec_20200118/"
    AllVideos_dir = "/share/caiyuanqiang/VideoSet/"
   
    # 'Cls10_YL','Cls11_YS','Cls12_QG','Cls13_FCJJ','Cls14_CY',
    # 'Cls7_YX','Cls8_XJ','Cls9_XYH','Cls6_XW'  
    # ["Cls1_ZYDS","Cls2_ECY","Cls3_TY","Cls4_MRMX","Cls5_PP"]
    ClsName_list = [
#         "Cls1_ZYDS","Cls2_ECY","Cls3_TY","Cls4_MRMX","Cls5_PP"
#         "Cls6_XW","Cls7_YX","Cls8_XJ","Cls9_XYH","Cls10_YL"
#         "Cls11_YS","Cls12_QG","Cls13_FCJJ","Cls14_CY","Cls15_SY"
#         "Cls16_ZWH","Cls17_JY","Cls18_LX","Cls19_SS","Cls20_XY"
#         "Cls21_MY","Cls22_QC","Cls23_HW","Cls24_YY","Cls25_DJ"
        "Cls26_KJ","Cls27_KP","Cls28_MY","Cls29_MZ","Cls30_WD"
    
    ]
    

    for Cls_name in ClsName_list:
        print("Class ==> {}".format(Cls_name))
        ClsVideo_dir = os.path.join(AllVideos_dir, Cls_name)
        VideoPath_list = get_file_path_list(ClsVideo_dir, ['.mp4'])

        ClsFrames_dir = ClsVideo_dir + "_Frames"
        ClsFramesDir_list = []
        ClsSampleGtTxtFrames_dir = os.path.join(AllFrames_dir, Cls_name)+"_SampleFrames"
        ClsSampleGtTxtFramesDir_list = []

        for video_path in VideoPath_list:
            video_name = os.path.split(video_path)[1]
            video_pre = os.path.splitext(video_name)[0]
            ClsFramesDir_list.append( os.path.join(ClsFrames_dir, video_pre) )
            ClsSampleGtTxtFramesDir_list.append( os.path.join(ClsSampleGtTxtFrames_dir, video_pre))

        for idx, ClsFrames_path in enumerate(ClsFramesDir_list):
            print("Video ==> {}".format(ClsFrames_path))
            ClsSampleGtTxtFrames_path = ClsSampleGtTxtFramesDir_list[idx]
            if not os.path.exists(ClsSampleGtTxtFrames_path):
                print("===== {} =====".format(ClsSampleGtTxtFrames_path))
                continue
            RecoveryVideoAnnotations(SampleAnno_dir=ClsSampleGtTxtFrames_path, Frames_dir=ClsFrames_path)


if __name__ == "__main__":
    print("Hello CAI")
    # parse()
    # BatchParse()
    # RecoveryVideoAnnotations()
    BatchRecoveryVideoAnnotations()


