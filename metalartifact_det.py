# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 10:18:09 2026

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 10:12:52 2026

@author: Administrator
"""

import cv2
import numpy as np


def otsu_on_bright_region(img, min_val=220):
    """
    只在亮區內做 Otsu
    img: grayscale image (0~255)
    min_val: 僅考慮大於此值的亮區
    """
    bright_pixels = img[img > min_val]

    if len(bright_pixels) == 0:
        return min_val

    hist = np.bincount(bright_pixels, minlength=256).astype(np.float64)

    total = bright_pixels.size
    values = np.arange(256)
    sum_total = np.dot(values, hist)

    sum_bg = 0.0
    weight_bg = 0.0
    max_var = -1.0
    best_thresh = min_val

    for t in range(min_val + 1, 256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue

        weight_fg = total - weight_bg
        if weight_fg == 0:
            break

        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg

        between_var = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2

        if between_var > max_var:
            max_var = between_var
            best_thresh = t

    return best_thresh


def detect_metal_clips(binary_img, area_min=20, area_max=2000):
    """
    從二值圖中抓金屬夾
    條件:
    - 小面積
    - 位於影像上半部
    - 靠左右兩側
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_img, connectivity=8
    )

    h, w = binary_img.shape
    detected_boxes = []

    for i in range(1, num_labels):  # 0 是背景
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        cx, cy = centroids[i]

        aspect_ratio = bw / float(bh) if bh > 0 else 0

        is_small_object = area_min < area < area_max
        in_upper_region = cy < h * 0.35
        near_left_or_right = (cx < w * 0.35) or (cx > w * 0.65)
        reasonable_shape = bw < 100 and bh < 100
        good_ratio = 0.3 < aspect_ratio < 3.5

        if (
            is_small_object
            and in_upper_region
            and near_left_or_right
            and reasonable_shape
            and good_ratio
        ):
            detected_boxes.append((x, y, bw, bh, area, cx, cy))

    return detected_boxes


if __name__ == "__main__":

    # ===== 1. 讀取影像 =====
    input_path = r"C:\RK11318466630007.jpg"
    output_binary_path = r"C:\output_binary.png"
    output_detect_path = r"C:\output_detect.png"

    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("影像讀取失敗，請檢查路徑是否正確。")
        raise SystemExit

    # ===== 2. Otsu on bright region =====
    thresh = otsu_on_bright_region(img, min_val=220)
    print("Otsu threshold (bright region) =", thresh)

    # ===== 3. 二值化 =====
    binary = np.where(img >= thresh, 255, 0).astype(np.uint8)
    cv2.imwrite(output_binary_path, binary)

    # ===== 4. 金屬夾偵測 =====
    boxes = detect_metal_clips(binary, area_min=20, area_max=2000)

    # ===== 5. 畫框 =====
    result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for idx, (x, y, bw, bh, area, cx, cy) in enumerate(boxes, start=1):
        cv2.rectangle(result, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
        cv2.putText(
            result,
            f"metal_clip_{idx}",
            (x, max(y - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )
        print(
            f"metal_clip_{idx}: x={x}, y={y}, w={bw}, h={bh}, area={area}, "
            f"cx={cx:.1f}, cy={cy:.1f}"
        )

    print("Detected metal clips:", len(boxes))

    # ===== 6. 存檔 =====
    cv2.imwrite(output_detect_path, result)

    # ===== 7. 顯示 =====
    cv2.imshow("Binary", binary)
    cv2.imshow("Metal Clip Detection", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()