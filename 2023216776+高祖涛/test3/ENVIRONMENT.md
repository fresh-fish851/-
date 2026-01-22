# 环境配置（test3）

## 1) Python 与虚拟环境
- 推荐 Python 3.11。
- 为加分项，将虚拟环境命名为 `gzt`。

Conda（CPU 版 PyTorch）：
```bash
conda create -n gzt python=3.11 pytorch torchvision torchaudio cpuonly -c pytorch
conda activate gzt
pip install pillow numpy
```

Conda（如需 GPU，仅示例，按实际 CUDA 版本调整 `cudatoolkit`）：
```bash
conda create -n gzt python=3.11 pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch -c nvidia
conda activate gzt
pip install pillow numpy
```

Pip 方案（CPU）：
```bash
python -m venv gzt
gzt\\Scripts\\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install pillow numpy
```

## 2) 依赖列表（最小）
- torch, torchvision, torchaudio（CPU 或 GPU 对应版本）
- pillow
- numpy

## 3) 运行提示
- 切换到 `test3/` 目录。
- 训练模型（保存至 `models/gzt_mnist_cnn.pth`）：  
  ```bash
  python test3.py --train --epochs 5 --batch-size 128 --lr 1e-3
  ```
- 预测学号照片：  
  ```bash
  python test3.py --predict path/to/your_id.png --expected-len 10
  ```
- 如果背景是白色，脚本会自动反色；如分割位数不对，先裁剪或调整 `expected-len`。
