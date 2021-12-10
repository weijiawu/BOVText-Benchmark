#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-08-09 14:39
# @Author  : lzneu
# @Site    : 

import numpy as np
from shapely.geometry import Polygon


def intersection(g, p):
    # 取g,p中的几何体信息组成多边形
    g = np.array(g)
    p = np.array(p)

    g = Polygon(g[:8].reshape((4, 2)))
    p = Polygon(p[:8].reshape((4, 2)))

    # 判断g,p是否为有效的多边形几何体
    if not g.is_valid or not p.is_valid:
        return 0

    # 取两个几何体的交集和并集
    inter = Polygon(g).intersection(Polygon(p)).area
    union = g.area + p.area - inter
    if union == 0:
        return 0
    else:
        return inter / union


def weighted_merge(g, p):
    # 取g,p两个几何体的加权（权重根据对应的检测得分计算得到）
    g[:8] = (g[8] * g[:8] + p[8] * p[:8]) / (g[8] + p[8])

    # 合并后的几何体的得分为两个几何体得分的总和
    g[8] = (g[8] + p[8])
    return g


def standard_nms(S, thres):
    # 标准NMS
    order = np.argsort(S[:, 8])[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ovr = np.array([intersection(S[i], S[t]) for t in order[1:]])
        inds = np.where(ovr <= thres)[0]
        order = order[inds + 1]

    return S[keep]


def nms_locality(polys, thres=0.3):
    '''
    locality aware nms of EAST
    :param polys: a N*9 numpy array. first 8 coordinates, then prob
    :return: boxes after nms
    '''
    S = []  # 合并后的几何体集合
    p = None  # 合并后的几何体
    polys = np.array(polys)
    for g in polys:
        if p is not None and intersection(g, p) > thres:  # 若两个几何体的相交面积大于指定的阈值，则进行合并
            p = weighted_merge(g, p)
        else:  # 反之，则保留当前的几何体
            if p is not None:
                S.append(p)
            p = g
    if p is not None:
        S.append(p)
    if len(S) == 0:
        return np.array([])
    return standard_nms(np.array(S), thres)


if __name__ == '__main__':

    print(Polygon(np.array([[343, 350], [448, 135],
                            [474, 143], [369, 359]])).area)
