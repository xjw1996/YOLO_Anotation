# http://hslpicker.com
import os
import time

import cv2
import numpy as np


label_index = "03"
label_name = "jacket"
label_plus = [1.0, 1.0, 1.1]  # jacket
dir_path = 'dataset/' + label_name + '/'

label_index2 = "03"
label_name2 = "jacket_person"

label_plus2 = [1.6, 1.0, 1.5]  # jacket_person
dir_path2 = 'dataset/' + label_name2 + '/'


cap = cv2.VideoCapture('person_outdoor.webm')
count = 0

# 定义边缘向外补偿

# 定义色彩范围
color_upper = np.array([20, 240, 255])
color_lower = np.array([0, 60, 160])

color_upper2 = np.array([180, 240, 255])
color_lower2 = np.array([175, 60, 160])

cap.set(cv2.CAP_PROP_POS_MSEC, 1000 * 531)

cap_alltime = cap.get(7)/cap.get(5)

def click_image(event, x, y, flags, param):
    if event==cv2.EVENT_LBUTTONDOWN:#事件=点击鼠标左键
        print("HSV is", img_hsv[y, x])

def run():
    # 读取图片
    # img_origin = cv2.imread('img/cc.jpg')

    # ret, img_origin = capture.read()
    global img, img_hsv, img_origin

    success, img_origin = cap.read()
    img_time = cap.get(0)/1000

    img = img_origin.copy()

    img_width = int(img.shape[1]*0.5)
    img_height = int(img.shape[0]*0.5)
    # img = cv2.flip(img, 1)
    img = cv2.resize(img, (img_width, img_height))

    # 筛选颜色
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    img_range = cv2.inRange(img_hsv, color_lower, color_upper)
    img_range += cv2.inRange(img_hsv, color_lower2, color_upper2)

    # 查找边缘 找出面积最大轮廓
    contours, hierarchy = cv2.findContours(
        img_range, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        cv2.imshow("image", img)
        return 1

    max_contour = contours[0]
    max_area = 0
    max_contour2 = contours[0]
    max_area2 = 0

    for contour_one in contours:
        area = cv2.contourArea(contour_one)
        if (area > max_area):
            max_contour2 = max_contour
            max_area2 = max_area
            max_contour = contour_one
            max_area = area
        elif (area > max_area2):
            max_contour2 = contour_one
            max_area2 = area

    if max_area > 0 and (max_area-max_area2)/max_area < 0.5:
        max_contour = np.vstack((max_contour, max_contour2))
    img = cv2.drawContours(img, [max_contour], 0, (0, 255, 0), 3)

    # 找出轮廓凸包
    # print(max_contour)
    # max_contour = np.append(max_contour, [[500, 200]])
    convex = cv2.convexHull(max_contour)

    img = cv2.drawContours(img, [convex], 0, (0, 255, 0), 3)

    # jacket
    x, y, w, h = cv2.boundingRect(convex)
    diff_h = int(h*label_plus[0]-h)
    y -= diff_h
    h += diff_h+int(h*label_plus[1]-h)
    diff_w = int((w*label_plus[2]-w)/2)
    x -= diff_w
    w += diff_w*2
    img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 3)
    img1 = img.copy()

    # jacket_person
    x2, y2, w2, h2 = cv2.boundingRect(convex)
    diff_h2 = int(h2*label_plus2[0]-h2)
    y2 -= diff_h2
    h2 += diff_h2+int(h2*label_plus2[1]-h2)
    diff_w2 = int((w2*label_plus2[2]-w2)/2)
    x2 -= diff_w2
    w2 += diff_w2*2
    img = cv2.rectangle(img, (x2, y2), (x2+w2, y2+h2), (255, 0, 255), 3)

    global count, timestamp
    count_all = 0
    try:
        with open(dir_path + 'total.txt', 'r') as f:
            for index, line in enumerate(f):
                count_all += 1
    except OSError:
        pass
    cv2.putText(img, "count : " + str(count) + '  all: '+str(count_all), (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 255), 2)
    cv2.putText(img, "time : " + '%.2f' % img_time + ' / ' + '%.2f' % cap_alltime, (20, 500),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)
    cv2.imshow("image", img)

    if cv2.waitKey(1) == 32:
        keycode = cv2.waitKey(0)
        if keycode == 100 and count > 0:  # d 按键
            # jacket
            try:
                os.remove(dir_path + 'labels/' + str(timestamp) + '.txt')
                os.remove(dir_path + 'images/' + str(timestamp) + '.jpg')
                os.remove(dir_path + 'annotations/' + str(timestamp) + '.jpg')

                lines = [l for l in open(dir_path + 'total.txt', "r") if l.find(
                    dir_path + 'images/' + str(timestamp) + '.jpg') == -1]
                with open(dir_path + 'total.txt', 'w') as f:
                    f.writelines(lines)
            except OSError:
                pass

            # jacket_person
            try:
                os.remove(dir_path2 + 'labels/' + str(timestamp) + '.txt')
                os.remove(dir_path2 + 'images/' + str(timestamp) + '.jpg')
                os.remove(dir_path2 + 'annotations/' + str(timestamp) + '.jpg')

                lines = [l for l in open(dir_path2 + 'total.txt', "r") if l.find(
                    dir_path2 + 'images/' + str(timestamp) + '.jpg') == -1]
                with open(dir_path2 + 'total.txt', 'w') as f:
                    f.writelines(lines)
            except OSError:
                pass

            # all
            try:
                cv2.putText(img, "deleted file : " + str(timestamp) + '.jpg', (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2)
                cv2.imshow("image", img)
                cv2.waitKey(300)
            except OSError:
                pass

        elif keycode == 99:  # c 按键
            timestamp = int(round(time.time() * 1000))

            # jacket
            if not os.path.exists(dir_path + 'labels/'):
                os.makedirs(dir_path + 'labels/')
            if not os.path.exists(dir_path + 'images/'):
                os.makedirs(dir_path + 'images/')
            if not os.path.exists(dir_path + 'annotations/'):
                os.makedirs(dir_path + 'annotations/')
            with open(dir_path + 'labels/' + str(timestamp) + '.txt', 'w+') as f:
                f.write(
                    label_index + ' ' +
                    str((x2 + w2/2) / img_width) + ' ' +
                    str((y2 + h2/2) / img_height) + ' ' +
                    str(w2 / img_width) + ' ' +
                    str(h2 / img_height)
                )
            with open(dir_path + 'total.txt', 'a+') as f:
                f.write(
                    dir_path + 'images/' + str(timestamp) + '.jpg\n'
                )
            cv2.imwrite(dir_path + 'images/' +
                        str(timestamp) + '.jpg', img_origin)
            cv2.imwrite(dir_path + 'annotations/' +
                        str(timestamp) + '.jpg', img1)

            # jacket_person
            if not os.path.exists(dir_path2 + 'labels/'):
                os.makedirs(dir_path2 + 'labels/')
            if not os.path.exists(dir_path2 + 'images/'):
                os.makedirs(dir_path2 + 'images/')
            if not os.path.exists(dir_path2 + 'annotations/'):
                os.makedirs(dir_path2 + 'annotations/')
            with open(dir_path2 + 'labels/' + str(timestamp) + '.txt', 'w+') as f:
                f.write(
                    label_index2 + ' ' +
                    str((x + w/2) / img_width) + ' ' +
                    str((y + h/2) / img_height) + ' ' +
                    str(w / img_width) + ' ' +
                    str(h / img_height)
                )
            with open(dir_path2 + 'total.txt', 'a+') as f:
                f.write(
                    dir_path2 + 'images/' + str(timestamp) + '.jpg\n'
                )
            cv2.imwrite(dir_path2 + 'images/' +
                        str(timestamp) + '.jpg', img_origin)
            cv2.imwrite(dir_path2 + 'annotations/' +
                        str(timestamp) + '.jpg', img)

            # all
            cv2.putText(img, "save to : " + str(timestamp) + '.jpg', (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2)
            cv2.imshow("image", img)
            cv2.waitKey(300)
            count += 1
    return 1


if __name__ == "__main__":
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_image)

    while (run()):
        continue
