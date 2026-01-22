# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Tuple
from pathlib import Path
import cv2

# (x1,y1,x2,y2,score,label,blue_ratio)
Labeled = Tuple[int,int,int,int,float,str,float]

def draw_labeled_boxes(img_bgr, labeled: List[Labeled]):
    out = img_bgr.copy()
    for (x1,y1,x2,y2,score,label,blue_ratio) in labeled:
        if label == "共享单车":
            color = (255, 0, 0)  # BGR: 蓝框
        else:
            color = (0, 255, 0)  # 绿框
        cv2.rectangle(out, (x1,y1), (x2,y2), color, 2)
        text = f"{score:.2f}"
        y_text = y1 - 8 if y1 - 8 > 10 else y1 + 18
        cv2.putText(out, text, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out

def save_pred_txt(path: Path, labeled: List[Labeled], hw):
    """
    保存为简单文本：label x1 y1 x2 y2 score blue_ratio
    """
    h, w = hw
    lines = []
    for (x1,y1,x2,y2,score,label,blue_ratio) in labeled:
        lines.append(f"{label}\t{x1}\t{y1}\t{x2}\t{y2}\t{score:.4f}\t{blue_ratio:.4f}")
    path.write_text("\n".join(lines), encoding="utf-8")
