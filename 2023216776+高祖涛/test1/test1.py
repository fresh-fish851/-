
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


img_path = "./test1/input.jpg"
save_dir = "./test1"   
os.makedirs(save_dir, exist_ok=True)


img = cv2.imread(img_path)
if img is None:
    raise ValueError("图像读取失败")
# img = cv2.resize(img, (256, 256))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h, w = gray.shape


def convolution(img, kernel):
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    output = np.zeros_like(img, dtype=np.float32)

    for i in range(h):
        for j in range(w):
            region = padded[i:i+kh, j:j+kw]
            output[i, j] = np.sum(region * kernel)

    return output


sobel_x_kernel = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

sobel_y_kernel = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
])

gx = convolution(gray, sobel_x_kernel)
gy = convolution(gray, sobel_y_kernel)

sobel = np.sqrt(gx**2 + gy**2)
sobel = np.clip(sobel, 0, 255).astype(np.uint8)

cv2.imwrite(os.path.join(save_dir, "sobel_result.jpg"), sobel)


custom_kernel = np.array([
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]
])

custom_filtered = convolution(gray, custom_kernel)
custom_filtered = np.clip(custom_filtered, 0, 255).astype(np.uint8)

cv2.imwrite(os.path.join(save_dir, "custom_kernel_result.jpg"), custom_filtered)


hist = np.zeros((3, 256), dtype=np.int32)

for c in range(3):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pixel = img[i, j, c]
            hist[c, pixel] += 1

# 可视化
colors = ['b', 'g', 'r']
plt.figure()
for c in range(3):
    plt.plot(hist[c], color=colors[c])
plt.title("Color Histogram")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.savefig(os.path.join(save_dir, "color_histogram.png"))
plt.close()


levels = 16
gray_q = (gray / (256 / levels)).astype(np.uint8)

glcm = np.zeros((levels, levels), dtype=np.float32)

for i in range(h):
    for j in range(w - 1):  # 0° 方向，相邻像素
        p = gray_q[i, j]
        q = gray_q[i, j + 1]
        glcm[p, q] += 1

glcm /= np.sum(glcm)

# 纹理特征计算
contrast = 0
energy = 0
homogeneity = 0

for i in range(levels):
    for j in range(levels):
        contrast += (i - j) ** 2 * glcm[i, j]
        energy += glcm[i, j] ** 2
        homogeneity += glcm[i, j] / (1 + abs(i - j))

texture_features = {
    "contrast": contrast,
    "energy": energy,
    "homogeneity": homogeneity
}

np.save(os.path.join(save_dir, "texture_features.npy"), texture_features)

print("实验完成")
