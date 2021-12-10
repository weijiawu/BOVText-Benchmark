import os
import json
import cv2
from tqdm import tqdm
import numpy as np
import math
root = "/share/wuweijia/Data/ICDAR2015/train_gts"


cha = {}
for i in "abcdefghijklmnopqrstuvwxyz":
    cha[i] = 0
sum1 = 0
for i in os.listdir(root):
    label_path = os.path.join(root,i)
    
    with open(label_path, encoding='utf-8', mode='r') as f:
        for line in f.readlines():
            params = line.strip().strip('\ufeff').strip('\xef\xbb\xbf').split(',')
            content = params[8]
            for cc in content:
                if cc in cha:
                    cha[cc]+=1  
                    sum1+=1
                
print(cha)
print(sum1)
for i in cha:
    
    cha[i] = cha[i]/sum1
print(cha)