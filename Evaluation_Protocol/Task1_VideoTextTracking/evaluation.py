# Metrics for multiple object text tracker benchmarking.
# https://github.com/weijiawu/MMVText-Benchmark

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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

def parse_args():
    """Defines and parses command-line arguments."""
    parser = argparse.ArgumentParser(description="evaluation on MMVText", formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--groundtruths', type=str,default='/share/wuweijia/Data/MMVText/train/annotation'
                        , help='Directory containing ground truth files.')
    parser.add_argument('--tests', type=str,default='/home/guoxiaofeng/.jupyter/wuweijia/VideoTextSpotting/task123/MMVText',
                        help='Directory containing tracker result files')
    parser.add_argument('--log', type=str, help='a place to record result and outputfile of mistakes', default='')
    parser.add_argument('--loglevel', type=str, help='Log level', default='info')
    parser.add_argument('--fmt', type=str, help='Data format', default='mot15-2D')
    parser.add_argument('--solver', type=str, default="lap", help='LAP solver to use')
    parser.add_argument('--skip', type=int, default=0, help='skip frames n means choosing one frame for every (n+1) frames')
    parser.add_argument('--iou', type=float, default=0.5, help='special IoU threshold requirement for small targets')
    return parser.parse_args()


def iou_matrix_polygen(objs, hyps, max_iou=1.):
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
    for row in range(m):
        for col in range(n):
            iou = calculate_iou_polygen(objs[row], hyps[col])
            dist = 1-iou
#             dist = iou
            # 更新到iou_mat
            if dist > max_iou:
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
            name = name.split("_")[0]+"_"+name.split("_")[1] + "_" + "/" + name.split("_")[2]
            gt_filename = os.path.join(self.data_root,name) 
            
        self.gt_frame_dict = read_results(gt_filename, self.data_type, is_gt=True)  # results_dict[fid] = [(tlwh, target_id, score)]
        self.gt_ignore_frame_dict = read_results(gt_filename, self.data_type, is_ignore=True)


    def reset_accumulator(self):
        self.acc = mm.MOTAccumulator(auto_id=True)

    def eval_frame(self, frame_id, trk_tlwhs, trk_ids, rtn_events=False):
        # results
        trk_tlwhs = np.copy(trk_tlwhs)
        trk_ids = np.copy(trk_ids)

        gt_objs = self.gt_frame_dict[frame_id]

        gts = []
        ids = []
        for gt in gt_objs:

            gts.append(gt[:8])
            ids.append(gt[-1])

        gt_objs = gts
        gt_objs = np.array(gt_objs,dtype=np.int32)
        ids = np.array(ids,dtype=np.int32)
        
        
        if np.size(gt_objs) != 0:
            gt_tlwhs = gt_objs
            gt_ids = ids
        else:
            gt_tlwhs = gt_objs
            gt_ids = ids
        
        iou_distance = iou_matrix_polygen(gt_tlwhs, trk_tlwhs, max_iou=0.5)
        
        # acc
        self.acc.update(gt_ids, trk_ids, iou_distance)  

        if rtn_events and iou_distance.size > 0 and hasattr(self.acc, 'last_mot_events'):
            events = self.acc.last_mot_events  
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
                trk_objs = np.array(trk_objs,dtype=np.int32)
                trk_tlwhs = trk_objs[:,:-1]
                trk_ids = trk_objs[:,-1]
            else:
                trk_tlwhs = np.array([])
                trk_ids = np.array([])
            

            # trk_tlwhs， box的坐标
            # trk_ids：  id
#             print(trk_tlwhs, trk_ids)
#             assert False
            self.eval_frame(str(frame_id), trk_tlwhs, trk_ids, rtn_events=False)

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
    args = parse_args()
    
    loglevel = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(loglevel, int):
        raise ValueError('Invalid log level: {} '.format(args.loglevel))
    logging.basicConfig(level=loglevel, format='%(asctime)s %(levelname)s - %(message)s', datefmt='%I:%M:%S')

    if args.solver:
        mm.lap.default_solver = args.solver
        mm.lap.default_solver = 'lap'

    data_type = 'text'

    seqs = os.listdir(args.tests)
            
    accs = []
    for seq in tqdm(seqs): # tqdm(seqs):
        # eval
        result_path = os.path.join(args.tests, seq)
        evaluator = Evaluator(args.groundtruths, seq, data_type)
        accs.append(evaluator.eval_file(result_path))
        
    # metric names
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
#         namemap={'mota': 'MOTA', 'motp' : 'MOTP'}
        namemap=mm.io.motchallenge_metric_names
    )
    
    print(strsummary)
    return summary['mota']['OVERALL']
    # Evaluator.save_summary(summary, os.path.join(result_root, 'summary_{}.xlsx'.format(exp_name)))

if __name__ == '__main__':
    main()