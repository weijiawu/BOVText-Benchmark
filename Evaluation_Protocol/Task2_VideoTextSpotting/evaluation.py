# Metrics for multiple object text tracker benchmarking.
# https://github.com/weijiawu/MMVText-Benchmark

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import re
import argparse
import os
import numpy as np
import copy
import motmetrics as mm
import logging
from tqdm import  tqdm
from tracking_utils.io import read_results, unzip_objs
from shapely.geometry import Polygon, MultiPoint
from motmetrics import math_util
from collections import OrderedDict
import io
import Levenshtein
def parse_args():
    """Defines and parses command-line arguments."""
    parser = argparse.ArgumentParser(description="""
Compute metrics for trackers using MOTChallenge ground-truth data with data preprocess.

Files
-----
All file content, ground truth and test files, have to comply with the
format described in

Milan, Anton, et al.
"Mot16: A benchmark for multi-object tracking."
arXiv preprint arXiv:1603.00831 (2016).
https://motchallenge.net/

Structure
---------

Layout for ground truth data
    <GT_ROOT>/<SEQUENCE_1>/gt_video_1.json
    <GT_ROOT>/<SEQUENCE_2>/gt_video_2.json
    ...

Layout for test data
    <TEST_ROOT>/<SEQUENCE_1>/video_1.json
    <TEST_ROOT>/<SEQUENCE_2>/video_2.json
    ...

Sequences of ground truth and test will be matched according to the `<SEQUENCE_X>`
string in the seqmap.""", formatter_class=argparse.RawTextHelpFormatter)
    
    parser = argparse.ArgumentParser(description="evaluation on MMVText", formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--groundtruths', type=str,default='./Test/Annotation'
                        , help='Directory containing ground truth files.')
    parser.add_argument('--tests', type=str,default='./BOVText_spotting',
                        help='Directory containing tracker result files')
    parser.add_argument('--log', type=str, help='a place to record result and outputfile of mistakes', default='')
    parser.add_argument('--loglevel', type=str, help='Log level', default='info')
    parser.add_argument('--fmt', type=str, help='Data format', default='mot15-2D')
    parser.add_argument('--solver', type=str, default="lap", help='LAP solver to use')
    parser.add_argument('--skip', type=int, default=0, help='skip frames n means choosing one frame for every (n+1) frames')
    parser.add_argument('--iou', type=float, default=0.5, help='special IoU threshold requirement for small targets')
    return parser.parse_args()


def cal_similarity(string1, string2):
    if string1 == "" and string2 == "":
        return 1.0
    # TODO 确定是否保留，当字符串差1个字符的时候，也算对
    if Levenshtein.distance(string1, string2) == 1 :
        return 0.95
    return 1 - Levenshtein.distance(string1, string2) / max(len(string1), len(string2))
    
def iou_matrix_polygen(objs, gt_transcription, hyps, transcription, max_iou=1.,max_similarity=0.9):
    if np.size(objs) == 0 or np.size(hyps) == 0:
        return np.empty((0, 0))

    objs = np.asfarray(objs)  # m
    hyps = np.asfarray(hyps)  # n
    m = objs.shape[0]
    n = hyps.shape[0]
    # 初始化一个m*n的矩阵
    dist_mat = np.zeros((m, n))
    
#     print(objs)
    assert objs.shape[1] == 8
    assert hyps.shape[1] == 8
    # 开始计算
    for x,row in enumerate(range(m)):
        for y,col in enumerate(range(n)):
            iou = calculate_iou_polygen(objs[row], hyps[col])
            dist = iou
            
            gt_trans = re.sub(u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a])","",gt_transcription[row]).lower()
            hyps_trans = re.sub(u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a])","",transcription[col]).lower()
            
#             gt_trans = gt_transcription[row]
#             hyps_trans = transcription[col]
            
#             if dist>0.8:
#                 with open('./data.txt','a') as f:    #设置文件对象
#                     f.write(gt_trans)
#                     f.write(",")
#                     f.write(hyps_trans)
#                     f.write("\n")
#                     if hyps_trans == "###":
#                         print(hyps[col])
                    #将字符串写入文件中
            
            # 更新到iou_mat
            if dist < max_iou or cal_similarity(gt_trans,hyps_trans)<0.9:
                dist = np.nan

            dist_mat[row][col] = dist
    return dist_mat

def calculate_iou_polygen(bbox1, bbox2):
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

class Evaluator(object):
    #           SVTS/images/val   video_5_5  mot
    # data_root: label path
    # seq_name: video name
    # data type: "text"
    def __init__(self, data_root, seq_name, data_type):
        self.data_root = data_root
        self.seq_name = seq_name
        self.data_type = data_type
        self.load_annotations()
        self.reset_accumulator()

    def load_annotations(self):
        assert self.data_type in ('mot', 'text')
        if self.data_type == 'mot':
            gt_filename = os.path.join(self.data_root, self.seq_name, 'gt', 'gt.txt')
            
        else:
            name = self.seq_name
            if len(name.split("_"))==3:
                name = name.split("_")[0]+"_"+name.split("_")[1] + "/" + name.split("_")[2]
            else:
                name = name.split("_")[0]+"_"+name.split("_")[1] + "/" + name.split("_")[2]+"_"+name.split("_")[3]+"_" + name.split("_")[4]
            gt_filename = os.path.join(self.data_root,name) 

        self.gt_frame_dict = read_results(gt_filename, self.data_type, is_gt=True)


    def reset_accumulator(self):
        self.acc = mm.MOTAccumulator(auto_id=True)

    def eval_frame(self, frame_id, trk_tlwhs, trk_ids, trk_transcription, rtn_events=False):
        # results
        trk_tlwhs = np.copy(trk_tlwhs)
        trk_ids = np.copy(trk_ids)
        trk_transcription = np.copy(trk_transcription)

        gt_objs = self.gt_frame_dict[frame_id]
            
        gts = []
        ids = []
        transcription = []
        ignored = []
        for gt in gt_objs:
            if gt["ID_transcription"] == "###" or gt["ID_transcription"] == "#1":
                ignored.append(np.array(gt["points"],dtype=np.float))
            else:
                gts.append(np.array(gt["points"],dtype=np.float))
                ids.append(gt["ID"])
                transcription.append(gt["ID_transcription"])

        gt_objs = gts
        gt_objs = np.array(gt_objs,dtype=np.int32)
        ids = np.array(ids,dtype=np.int32)
        ignored = np.array(ignored,dtype=np.int32)
        
        
        if np.size(gt_objs) != 0:
            gt_tlwhs = gt_objs
            gt_ids = ids
            gt_transcription = transcription
        else:
            gt_tlwhs = gt_objs
            gt_ids = ids
            gt_transcription = transcription

        
        # filter 
        trk_tlwhs_ = []
        trk_ids_ = []
        trk_transcription_ = []
        
        for idx,box1 in enumerate(trk_tlwhs):
            flag = 0
            for box2 in ignored:
                iou = calculate_iou_polygen(box1, box2)
                if iou > 0.5:
                    flag=1
            if flag == 0:
                trk_tlwhs_.append(trk_tlwhs[idx])
                trk_ids_.append(trk_ids[idx])
                trk_transcription_.append(trk_transcription[idx])
                
        trk_tlwhs = trk_tlwhs_
        trk_ids = trk_ids_
        trk_transcription = trk_transcription_
        
        iou_distance = iou_matrix_polygen(gt_tlwhs,gt_transcription, trk_tlwhs, trk_transcription, max_iou=0.5, max_similarity=0.9) # 注意 这里是越小越好！！！
        
        # acc
        self.acc.update(gt_ids, trk_ids, iou_distance)  # 1 - iou 也就是交叠比< max_iou的设置为inf

        if rtn_events and iou_distance.size > 0 and hasattr(self.acc, 'last_mot_events'):
            events = self.acc.last_mot_events  # only supported by https://github.com/longcw/py-motmetrics
        else:
            events = None
        return events

    def eval_file(self, filename):
        self.reset_accumulator()
        result_frame_dict = read_results(filename, self.data_type, is_gt=False)

        for frame_id in range(len(self.gt_frame_dict)):
            frame_id += 1
            if str(frame_id) in result_frame_dict.keys():
                trk_objs = result_frame_dict[str(frame_id)]
                
                trk_tlwhs = []
                trk_ids = []
                trk_transcription = []
                for trk in trk_objs:
#                     print(trk)
                    trk_tlwhs.append(np.array(trk["points"],dtype=np.int32))
                    trk_ids.append(np.array(trk["ID"],dtype=np.int32))
                    try:
                        trk_transcription.append(trk["transcription"])
                    except:
                        trk_transcription.append("error")
                        print(trk)
            else:
                trk_tlwhs = np.array([])
                trk_ids = np.array([])
                trk_transcription = []
                
            self.eval_frame(str(frame_id), trk_tlwhs, trk_ids,trk_transcription, rtn_events=False)

        return self.acc

    @staticmethod
    def get_summary(accs, names, metrics=('mota','motp', 'num_switches', 'idp', 'idr', 'idf1', 'precision', 'recall')):
        names = copy.deepcopy(names)
        if metrics is None:
            metrics = mm.metrics.motchallenge_metrics
        metrics = copy.deepcopy(metrics)

        mh = mm.metrics.create()
        summary = mh.compute_many(
            accs,
            metrics=metrics,
            names=names,
            generate_overall=True
        )
        return summary

    @staticmethod
    def save_summary(summary, filename):
        import pandas as pd
        writer = pd.ExcelWriter(filename)
        summary.to_excel(writer)
        writer.save()
        
def main():
    class_list = ["Cls1_Livestreaming","Cls2_Cartoon","Cls3_Sports", "Cls4_Celebrity", "Cls5_Advertising"
          ,"Cls6_NewsReport", "Cls7_Game","Cls8_Comedy","Cls9_Activity","Cls10_Program"
          ,"Cls11_Movie","Cls12_Interview","Cls13_Introduction","Cls14_Talent","Cls15_Photograph"
          ,"Cls16_Government","Cls17_Speech","Cls18_Travel","Cls19_Fashion","Cls20_Campus"
          ,"Cls21_Vlog","Cls22_Driving","Cls23_International","Cls24_Fishery","Cls25_ShortVideo"
          ,"Cls26_Technology","Cls27_Education","Cls28_BeautyIndustry","Cls29_Makeup","Cls30_Dance","Cls31_Eating","Cls32_Unknown"]
    
    args = parse_args()
    
    loglevel = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(loglevel, int):
        raise ValueError('Invalid log level: {} '.format(args.loglevel))
    logging.basicConfig(level=loglevel, format='%(asctime)s %(levelname)s - %(message)s', datefmt='%I:%M:%S')

    if args.solver:
        mm.lap.default_solver = args.solver
        mm.lap.default_solver = 'lap'

    data_type = 'text'
    
    filter_seqs = []
    for cls in os.listdir(args.groundtruths):
        if cls == ".ipynb_checkpoints":
            continue
        video_name = os.path.join(args.groundtruths,cls)
        for video_ in os.listdir(video_name):
            if video_ == ".ipynb_checkpoints":
                continue
            filter_seqs.append(cls+"_"+video_)
    
    
    accs = []
    for seq in tqdm(filter_seqs): # tqdm(seqs):
        # eval
        
        result_path = os.path.join(args.tests, seq)
        evaluator = Evaluator(args.groundtruths, seq, data_type)
        accs.append(evaluator.eval_file(result_path))
        
    # metric names
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, filter_seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    
    print(strsummary)
    return summary['mota']['OVERALL']
    # Evaluator.save_summary(summary, os.path.join(result_root, 'summary_{}.xlsx'.format(exp_name)))

if __name__ == '__main__':
    main()