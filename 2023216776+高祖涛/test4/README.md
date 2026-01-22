# 实验四：校园共享单车检测（蓝色=共享单车）

本项目实现：
1) 基于 COCO 预训练检测器定位单车（bicycle）
2) 在检测框 ROI 内做蓝色占比统计：蓝色=共享单车；否则=普通单车

## 目录结构
- `src/` 代码
- `assets/` 放置模型文件（cfg/weights/coco.names）
- `outputs/` 输出结果
- `data/images/`（你本地创建）放测试图片

## 1. 环境安装
```bash
pip install -r requirements.txt
```

## 2. 准备模型文件（COCO）
你需要把以下 3 个文件放到 `assets/`：
- `yolov3.cfg`
- `yolov3.weights`
- `coco.names`

说明：可以使用 YOLOv4 / YOLOv4-tiny 等，只要 cfg/weights 匹配且类别文件是 COCO 即可。

## 3. 运行（单张或批量）
### 批量：输入为文件夹
```bash
python src/main.py --input data/images --output outputs --backend opencv-yolo   --yolo_cfg assets/yolov3.cfg --yolo_weights assets/yolov3.weights --coco_names assets/coco.names   --save_txt
```

### 单张：输入为图片
```bash
python src/main.py --input data/images/test.jpg --output outputs --save_txt
```

## 4. 调参建议
- `--conf_thres`：检测置信度阈值（0.25~0.5）
- `--blue_ratio_thres`：蓝色占比阈值（0.10~0.25）
  - 光照强、车身蓝色明显：可设大一点（更严格）
  - 光照弱、蓝色区域较少：可设小一点（更宽松）

## 5. 输出
- `*_out.jpg`：绘制 bbox + 标签（共享单车/普通单车）
- `*_pred.txt`（可选）：每个检测框一行：label x1 y1 x2 y2 score blue_ratio
