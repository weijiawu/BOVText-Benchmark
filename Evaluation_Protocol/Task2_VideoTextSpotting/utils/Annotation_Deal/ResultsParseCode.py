#-*-coding:utf-8
import os
import math
import numpy as np
import cv2


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
            wline = ":".join(line)
            wline += "\n"
            fw.write(wline)

def parse(Anno_path=""):
    """
    {"id":7,"photoId":"KSText_ReDetRec_20200118/Cls10_YL_SampleFrames/41118204665/41118204665_000042.jpg_000042.jpg",
    "positionList":[{"extraContent":"妈","option":"背景文字","positionTuple":["16.656536:1232.0","72.0:1232.0","72.0:1287.0","16.656536:1287.0"],"frameNo":null},
    {"extraContent":"猫眼","option":"前景文字","positionTuple":["105.4914:68.47487","235.04224:68.47487","235.04224:128.6217","105.4914:128.6217"],"frameNo":null},
    {"extraContent":"剧综团","option":"前景文字","positionTuple":["77.73051:132.32306","272.0:132.32306","272.0:198.94725","77.73051:198.94725"],"frameNo":null},
    {"extraContent":"朱迅呐喊出心底的愿望","option":"前景文字","positionTuple":["49.0:632.0045","604.26215:632.0045","604.26215:696.778","49.0:696.778"],"frameNo":null},
    {"extraContent":"朱迅的呐喊太感人了","option":"前景文字","positionTuple":["217.46034:325.7183","857.81165:325.7183","857.81165:404.37186","217.46034:404.37186"],"frameNo":null}],
    "keywords":"默认分类","mmuLabelId":7}
    :param Anno_path:
    :return:
    """
    # Anno_path = "../Results_DetRec/5106_OCR12改框0129_EtXJbbAm_20210408230421.txt"
    Root_dir = "./"
    ID = 1
    ID_dic = {}
    with open(Anno_path, "r",encoding='utf-8') as fr:
        for line in fr.readlines():
#             print(line)
            line = line.replace("null","1")
            dic = eval(line)
            idx = dic['photoId'].rfind('_')
            img_path =dic['photoId'][:idx]  # KSText_ReDetRec_20200118/Cls10_YL_SampleFrames/41118204665/41118204665_000042.jpg
            GtTxt_dir = os.path.join(Root_dir, os.path.split(img_path)[0]).replace("./KSText_ReDetRec_20200118/NewVideo_Frames","./Annotation")
            if not os.path.exists(GtTxt_dir):
                os.makedirs(GtTxt_dir)
            GtTxt_path = os.path.join(Root_dir, os.path.splitext(img_path)[0] + ".txt").replace("./KSText_ReDetRec_20200118/NewVideo_Frames","./Annotation")
#             print(GtTxt_path)
#             assert False
            
            image_text_list = []
            for box_inf in dic['positionList']:
                # print(box_inf['extraContent'])
                extractContent = str(box_inf['extraContent'])
                # print(box_inf['option'])
                option = str(box_inf['option'])
                ID_str = str(box_inf['frameNo'])
                if ID_str in ID_dic.keys():
                    ID_one = ID_dic[ID_str]
                else:
                    ID_dic.update({ID_str:str(ID)})
                    ID_one = ID_dic[ID_str]
                    ID+=1
                    
                if len(box_inf['positionTuple']) == 0:
                    continue

                x1 = int(float(box_inf['positionTuple'][0].split(":")[0]))
                y1 = int(float(box_inf['positionTuple'][0].split(":")[1]))

                x2 = int(float(box_inf['positionTuple'][1].split(":")[0]))
                y2 = int(float(box_inf['positionTuple'][1].split(":")[1]))

                x3 = int(float(box_inf['positionTuple'][2].split(":")[0]))
                y3 = int(float(box_inf['positionTuple'][2].split(":")[1]))

                x4 = int(float(box_inf['positionTuple'][3].split(":")[0]))
                y4 = int(float(box_inf['positionTuple'][3].split(":")[1]))

                points = [x1,y1,x2,y2,x3,y3,x4,y4]
                points = [str(e) for e in points]
                points.append(ID_one)
                points.append(extractContent)
                points.append(option)
                image_text_list.append(points)

            write_4points(GtTxt_path, image_text_list)

def BatchParse():
    '''  '''
    Results_dir = "../Results_DetRec"
#     Anno_path_list = get_file_path_list(Results_dir, ['.txt'])
#     for Anno_path in Anno_path_list:
    parse("./origrinal_label.txt")

if __name__ == "__main__":
    print("Hello CAI")
    # parse()
    BatchParse()


