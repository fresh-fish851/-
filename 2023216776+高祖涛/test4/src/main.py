# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path
import cv2

from detector_opencv_yolo import detect_bicycles_opencv_yolo
from color_classifier import classify_bike_by_blue_ratio
from vis import draw_labeled_boxes, save_pred_txt

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def iter_images(input_path: Path):
    if input_path.is_file():
        if input_path.suffix.lower() in IMG_EXTS:
            yield input_path
        return
    for p in sorted(input_path.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True, help="输入图像路径（单张图片）或文件夹（批量）")
    ap.add_argument("--output", type=str, default="outputs", help="输出文件夹")
    ap.add_argument("--backend", type=str, default="opencv-yolo", choices=["opencv-yolo"],
                    help="检测后端：当前提供 OpenCV DNN + YOLOv3/YOLOv4 Darknet 权重")
    ap.add_argument("--yolo_cfg", type=str, default="assets/yolov3.cfg", help="Darknet cfg 路径")
    ap.add_argument("--yolo_weights", type=str, default="assets/yolov3.weights", help="Darknet weights 路径")
    ap.add_argument("--coco_names", type=str, default="assets/coco.names", help="COCO 类别名文件路径")
    ap.add_argument("--conf_thres", type=float, default=0.35, help="置信度阈值")
    ap.add_argument("--nms_thres", type=float, default=0.45, help="NMS 阈值")
    ap.add_argument("--blue_ratio_thres", type=float, default=0.18, help="蓝色占比阈值（越大越严格）")
    ap.add_argument("--save_txt", action="store_true", help="同时保存 txt 预测结果（bbox+label）")
    return ap.parse_args()

def main():
    args = parse_args()
    in_path = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in iter_images(in_path):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] 读取失败：{img_path}")
            continue

        # 1) 检测单车 bbox（COCO bicycle）
        boxes = detect_bicycles_opencv_yolo(
            img,
            yolo_cfg=args.yolo_cfg,
            yolo_weights=args.yolo_weights,
            coco_names=args.coco_names,
            conf_thres=args.conf_thres,
            nms_thres=args.nms_thres,
        )

        # 2) 颜色判别：蓝色=共享单车
        labeled = []
        for (x1, y1, x2, y2, score) in boxes:
            h = y2 - y1
            w = x2 - x1

            # 取中间区域（去掉上下左右各 25%）
            cx1 = x1 + int(0.25 * w)
            cx2 = x2 - int(0.25 * w)
            cy1 = y1 + int(0.25 * h)
            cy2 = y2 - int(0.25 * h)

            roi = img[cy1:cy2, cx1:cx2]
            label, blue_ratio = classify_bike_by_blue_ratio(roi)

            if label == "共享单车":
                labeled.append((x1, y1, x2, y2, score, label, blue_ratio))


        # 3) 可视化输出
        vis = draw_labeled_boxes(img, labeled)
        out_name = img_path.stem + "_out.jpg"
        out_img_path = out_dir / out_name
        cv2.imwrite(str(out_img_path), vis)
        print(f"[OK] {img_path.name} -> {out_img_path.name} (bikes={len(labeled)})")

        if args.save_txt:
            out_txt_path = out_dir / (img_path.stem + "_pred.txt")
            save_pred_txt(out_txt_path, labeled, img.shape[:2])

if __name__ == "__main__":
    main()
