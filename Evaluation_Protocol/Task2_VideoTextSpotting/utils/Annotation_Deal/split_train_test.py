import os
import numpy as np

root = "/share/wuweijia/Data/MMVText/train"
annotation_path = root + "/annotation"


train_list = []
test_list = []
for class_name in os.listdir(annotation_path):
    video_path = os.path.join(annotation_path,class_name)
    for idx,video_path_ in enumerate(os.listdir(video_path)):
        ratio = int(len(os.listdir(video_path))*0.6)
        if idx > ratio:
            keys = os.path.join(class_name,video_path_)
            test_list.append(keys)
        else:
            keys = os.path.join(class_name,video_path_)
            train_list.append(keys)
            
            
result = os.path.join(root,"train_list.txt")
with open(result, 'w') as f:
    for line in train_list:
        f.write(line)
        f.write("\n")
        

result = os.path.join(root,"test_list.txt")
with open(result, 'w') as f:
    for line in test_list:
        f.write(line)
        f.write("\n")