# http://hslpicker.com
import os
import time

import cv2
import numpy as np

capture = cv2.VideoCapture(0)

label_index = "02"
label_name = "can"
dir_path = 'dataset/' + label_name + '/'
cap = cv2. VideoCapture('can.mp4')
count = 0

# 定义边缘向外补偿
label_plus_onlytop = 0
# label_plus = np.array([1, 2])  # 便当
# label_plus = np.array([1.8, 1])  # 瓶子
label_plus = np.array([1.3, 1])  # can

# 定义色彩范围
color_upper = np.array([180, 200, 255])
color_lower = np.array([160, 30, 60])

white_upper = np.array([180, 255, 255])
white_lower = np.array([0, 250, 0])


# label_plus = np.array([1, 1.6])


def run():
    # 读取图片
    # img_origin = cv2.imread('img/cc.jpg')

    #ret, img_origin = capture.read()

    success, img_origin = cap.read()

    img = img_origin.copy()

    img_width = int(img.shape[1]*0.5)
    img_height = int(img.shape[0]*0.5)
    # img = cv2.flip(img, 1)
    img = cv2.resize(img, (img_width, img_height))

    # 筛选颜色
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    img_range = cv2.inRange(img_hsv, color_lower, color_upper)
    # img_range += cv2.inRange(img_hsv, white_lower, white_upper)

    # 查找边缘 找出面积最大轮廓
    contours, hierarchy = cv2.findContours(
        img_range, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        cv2.imshow("image", img)
        return 1

    max_contour = contours[0]
    max_area = 0
    for contour_one in contours:
        area = cv2.contourArea(contour_one)
        if (area > max_area):
            max_contour = contour_one
            max_area = area
    img = cv2.drawContours(img, [max_contour], 0, (0, 255, 0), 3)

    # 找出轮廓凸包
    # print(max_contour)
    # max_contour = np.append(max_contour, [[500, 200]])
    convex = cv2.convexHull(max_contour)

    img = cv2.drawContours(img, [convex], 0, (0, 255, 0), 3)

    # 最小斜矩形((978.2109375, 467.5702209472656), (515.8818359375, 323.0546875), -39.01634216308594)
    min_area_rect = cv2.minAreaRect(convex)
    # 最小斜矩形结果转换成轮廓点
    min_area_box = cv2.boxPoints(min_area_rect)

    cv2.drawContours(img, [np.int0(min_area_box)], 0, (255, 0, 255), 3)
    is_first_long = min_area_rect[1][0] > min_area_rect[1][1]

    min_area_rect_plus = list(min_area_rect)
    if is_first_long:
        min_area_rect_plus[1] = tuple(min_area_rect_plus[1] * label_plus)
    else:
        min_area_rect_plus[1] = tuple(min_area_rect_plus[1] * label_plus[::-1])

    min_area_box_plus = cv2.boxPoints(tuple(min_area_rect_plus))
    cv2.drawContours(img, [np.int0(min_area_box_plus)], 0, (255, 255, 0), 3)

    if label_plus_onlytop:
        if is_first_long:
            min_x = np.min(min_area_box.T[0])
            min_y = np.min(min_area_box_plus.T[1])
            max_x = np.max(min_area_box_plus.T[0])
            max_y = np.max(min_area_box.T[1])
        else:
            min_x = np.min(min_area_box_plus.T[0])
            min_y = np.min(min_area_box_plus.T[1])
            max_x = np.max(min_area_box.T[0])
            max_y = np.max(min_area_box.T[1])
    else:
        min_x = np.min(min_area_box_plus.T[0])
        min_y = np.min(min_area_box_plus.T[1])
        max_x = np.max(min_area_box_plus.T[0])
        max_y = np.max(min_area_box_plus.T[1])

    cv2.drawContours(img, [np.int0([[min_x, min_y], [min_x, max_y], [max_x, max_y], [max_x, min_y]])], 0, (0, 0, 255),
                     4)
    global count, timestamp
    count_all = 0
    with open(dir_path + 'total.txt', 'r') as f:
        for index, line in enumerate(f):
            count_all += 1
    cv2.putText(img, "count : " + str(count) + '  all: '+str(count_all), (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 255), 2)
    cv2.imshow("image", img)
    if cv2.waitKey(1) == 32:
        keycode = cv2.waitKey(0)
        if keycode == 100 and count > 0:  # d 按键
            try:
                os.remove(dir_path + 'labels/' + str(timestamp) + '.txt')
                os.remove(dir_path + 'images/' + str(timestamp) + '.jpg')
                os.remove(dir_path + 'annotations/' + str(timestamp) + '.jpg')

                lines = [l for l in open(dir_path + 'total.txt', "r") if l.find(
                    dir_path + 'images/' + str(timestamp) + '.jpg') == -1]
                with open(dir_path + 'total.txt', 'w') as f:
                    f.writelines(lines)
                cv2.putText(img, "deleted file : " + str(timestamp) + '.jpg', (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2)
                cv2.imshow("image", img)
                cv2.waitKey(300)
            except OSError:
                pass
        elif keycode == 99:  # c 按键
            timestamp = int(round(time.time() * 1000))
            if not os.path.exists(dir_path + 'labels/'):
                os.makedirs(dir_path + 'labels/')
            if not os.path.exists(dir_path + 'images/'):
                os.makedirs(dir_path + 'images/')
            if not os.path.exists(dir_path + 'annotations/'):
                os.makedirs(dir_path + 'annotations/')

            with open(dir_path + 'labels/' + str(timestamp) + '.txt', 'w+') as f:
                f.write(
                    label_index + ' ' +
                    str((max_x + min_x) / 2 / img_width) + ' ' +
                    str((max_y + min_y) / 2 / img_height) + ' ' +
                    str((max_x - min_x) / img_width) + ' ' +
                    str((max_y - min_y) / img_height)
                )
            with open(dir_path + 'total.txt', 'a+') as f:
                f.write(
                    dir_path + 'images/' + str(timestamp) + '.jpg\n'
                )

            cv2.imwrite(dir_path + 'images/' +
                        str(timestamp) + '.jpg', img_origin)
            cv2.imwrite(dir_path + 'annotations/' +
                        str(timestamp) + '.jpg', img)
            cv2.putText(img, "save to : " + str(timestamp) + '.jpg', (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2)
            cv2.imshow("image", img)
            cv2.waitKey(300)
            count += 1
    return 1


if __name__ == "__main__":
    cv2.namedWindow("image")
    while (run()):
        continue
