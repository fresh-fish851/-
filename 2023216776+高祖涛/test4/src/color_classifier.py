# -*- coding: utf-8 -*-
"""
颜色判别：蓝色=共享单车
做法：在检测框 ROI 内统计“蓝色像素占比”，超过阈值则判为共享单车。
为了减少背景/阴影影响，先用饱和度/亮度筛选“有效像素”，再计算蓝色比值。
"""
from __future__ import annotations
from typing import Tuple
import numpy as np
import cv2

def classify_bike_by_blue_ratio(roi_bgr, ratio_thres: float = 0.18) -> Tuple[str, float]:
    """
    返回 (label, blue_ratio)，label ∈ {"共享单车", "普通单车"}。
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return "普通单车", 0.0

    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

    # 蓝色范围（经验值，可在报告中说明通过实验调整）
    # H: 90~135 约覆盖常见蓝色；S/V 下限避免暗部、灰度区域误判
    lower_blue = np.array([90, 60, 60], dtype=np.uint8)
    upper_blue = np.array([135, 255, 255], dtype=np.uint8)
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # 有效像素：排除非常低饱和度/亮度的区域，减少背景干扰
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    valid_mask = ((s >= 40) & (v >= 40)).astype(np.uint8) * 255

    blue = (blue_mask > 0) & (valid_mask > 0)
    valid = (valid_mask > 0)

    valid_cnt = int(valid.sum())
    blue_cnt = int(blue.sum())

    if valid_cnt < 50:
        # ROI 太小/过暗时退化为全 ROI 统计
        blue_ratio = float((blue_mask > 0).sum()) / float(blue_mask.size)
    else:
        blue_ratio = float(blue_cnt) / float(valid_cnt)

    label = "共享单车" if blue_ratio >= ratio_thres else "普通单车"
    return label, blue_ratio
