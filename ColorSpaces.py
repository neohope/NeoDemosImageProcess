#!/usr/bin/python
# -*- coding:utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt


'''
加载并转换图像
OpenCV默认为BGR
matplotlib默认为RGB
'''
def loadimage(imgpath):
    # 读取原始图像

    img_BGR = cv2.imread(imgpath)

    # BGR转换为RGB
    img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)

    # 灰度化处理
    img_GRAY = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY)

    # BGR转HSV
    img_HSV = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2HSV)

    # BGR转YCrCb
    img_YCrCb = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)

    # BGR转HLS
    img_HLS = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2HLS)

    # BGR转XYZ
    img_XYZ = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2XYZ)

    # BGR转LAB
    img_LAB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2LAB)

    # BGR转YUV
    img_YUV = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YUV)

    return [img_BGR, img_RGB, img_GRAY, img_HSV, img_YCrCb,
            img_HLS, img_XYZ, img_LAB, img_YUV]


'''
展示结果
'''
def showImage(images):
    # 调用matplotlib显示处理结果
    titles = ['BGR', 'RGB', 'GRAY', 'HSV', 'YCrCb', 'HLS', 'XYZ', 'LAB', 'YUV']
    for i in range(9):
        plt.subplot(3, 3, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    imgpath = 'images/Pig_160X160.png'
    images = loadimage(imgpath)
    showImage(images)
