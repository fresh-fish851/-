import cv2
import numpy as np
import os


img_path = "./test2/road1.png"         
save_dir = "./test2"
os.makedirs(save_dir, exist_ok=True)


img = cv2.imread(img_path)
if img is None:
    raise ValueError("图像读取失败")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def region_of_interest(img):
    h, w = img.shape
    mask = np.zeros_like(img)

    polygon = np.array([[
        (int(0.10 * w), h),
        (int(0.15 * w), int(0.62 * h)),
        (int(0.95 * w), int(0.62 * h)),
        (int(0.99 * w), h)
    ]], dtype=np.int32)

    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(img, mask)


blur = cv2.GaussianBlur(gray, (5, 5), 0)

edges = cv2.Canny(blur, 70, 150)

edges = region_of_interest(edges)


lines = cv2.HoughLinesP(
    edges,
    rho=1,
    theta=np.pi / 180,
    threshold=40,
    minLineLength=40,
    maxLineGap=50
)


def filter_lines(lines):
    left_lines = []
    right_lines = []

    if lines is None:
        return left_lines, right_lines

    for line in lines:
        x1, y1, x2, y2 = line[0]

        if x2 == x1:
            continue

        slope = (y2 - y1) / (x2 - x1)
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        # 长度约束
        if length < 40:
            continue

        if  slope < -1:
            left_lines.append((x1, y1, x2, y2))
            print("OK")
        elif  slope > 1:
            right_lines.append((x1, y1, x2, y2))

    return left_lines, right_lines


def average_line(lines, img_height):
    if len(lines) == 0:
        return None

    slopes = []
    intercepts = []

    for x1, y1, x2, y2 in lines:
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        slopes.append(slope)
        intercepts.append(intercept)

    slope_avg = np.mean(slopes)
    intercept_avg = np.mean(intercepts)

    y1 = img_height
    y2 = int(img_height * 0.62)

    x1 = int((y1 - intercept_avg) / slope_avg)
    x2 = int((y2 - intercept_avg) / slope_avg)

    return x1, y1, x2, y2

left_lines, right_lines = filter_lines(lines)

left_lane = average_line(left_lines, img.shape[0])
right_lane = average_line(right_lines, img.shape[0])


lane_img = img.copy()

if left_lane is not None:
    cv2.line(
        lane_img,
        (left_lane[0], left_lane[1]),
        (left_lane[2], left_lane[3]),
        (0, 0, 255),
        6
    )

if right_lane is not None:
    cv2.line(
        lane_img,
        (right_lane[0], right_lane[1]),
        (right_lane[2], right_lane[3]),
        (0, 0, 255),
        6
    )


cv2.imwrite(os.path.join(save_dir, "edges.jpg"), edges)
cv2.imwrite(os.path.join(save_dir, "lane_detection_result.jpg"), lane_img)

print("实验二完成：最终车道线检测结果已保存")
