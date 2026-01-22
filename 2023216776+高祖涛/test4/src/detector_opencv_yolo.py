# -*- coding: utf-8 -*-
"""
OpenCV DNN + Darknet YOLO 检测 COCO bicycle。
支持 yolov3 / yolov4 / yolov4-tiny 等（只要 cfg + weights + coco.names 对应即可）。
"""
from __future__ import annotations
from typing import List, Tuple
import cv2
import numpy as np
from pathlib import Path

Box = Tuple[int, int, int, int, float]  # x1,y1,x2,y2,score

def _load_names(coco_names: str):
    names = Path(coco_names).read_text(encoding="utf-8", errors="ignore").splitlines()
    names = [n.strip() for n in names if n.strip()]
    return names

def _get_output_layer_names(net):
    layer_names = net.getLayerNames()
    try:
        # OpenCV 4.x: getUnconnectedOutLayers() returns 1-based indices
        out_layers = net.getUnconnectedOutLayers().flatten().tolist()
        return [layer_names[i - 1] for i in out_layers]
    except Exception:
        return net.getUnconnectedOutLayersNames()

def detect_bicycles_opencv_yolo(
    img_bgr,
    yolo_cfg: str,
    yolo_weights: str,
    coco_names: str,
    conf_thres: float = 0.35,
    nms_thres: float = 0.45,
    inp_size: int = 416,
) -> List[Box]:
  
    h, w = img_bgr.shape[:2]

    names = _load_names(coco_names)
    # COCO 中 bicycle 通常为 "bicycle"
    try:
        bicycle_id = names.index("bicycle")
    except ValueError:
        raise RuntimeError("coco.names 中未找到 'bicycle' 类别，请检查类别文件是否为 COCO。")

    net = cv2.dnn.readNetFromDarknet(str(yolo_cfg), str(yolo_weights))
    # 可按需切换推理后端/目标（CPU/GPU）
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    blob = cv2.dnn.blobFromImage(img_bgr, 1/255.0, (inp_size, inp_size), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(_get_output_layer_names(net))

    boxes = []
    confidences = []

    for out in outs:
        for det in out:
            # det: [center_x, center_y, width, height, obj_conf, class_scores...]
            scores = det[5:]
            class_id = int(np.argmax(scores))
            class_score = float(scores[class_id])
            obj_conf = float(det[4])
            conf = obj_conf * class_score

            if class_id != bicycle_id:
                continue
            if conf < conf_thres:
                continue

            cx, cy, bw, bh = det[0:4]
            cx *= w; cy *= h; bw *= w; bh *= h
            x = int(cx - bw / 2); y = int(cy - bh / 2)
            boxes.append([x, y, int(bw), int(bh)])
            confidences.append(conf)

    # NMS
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_thres, nms_thres)
    res: List[Box] = []
    if len(idxs) > 0:
        for i in idxs.flatten().tolist():
            x, y, bw, bh = boxes[i]
            x1 = max(0, x); y1 = max(0, y)
            x2 = min(w - 1, x + bw); y2 = min(h - 1, y + bh)
            res.append((x1, y1, x2, y2, float(confidences[i])))
    return res
