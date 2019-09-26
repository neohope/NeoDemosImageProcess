#!/usr/bin/python
# -*- coding:utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt


'''
加载并转换图像
'''
def loadimage(imgpath):

    img_BGR = cv2.imread(imgpath, cv2.IMREAD_UNCHANGED)
    print(img_BGR.shape)
    print(img_BGR.size)
    print(img_BGR.dtype)
    cv2.imshow("BGR", img_BGR)
    cv2.waitKey(0)

    #channels
    rows, cols, chn = img_BGR.shape
    zeros = np.zeros((rows, cols), dtype=img_BGR.dtype)
    #img_B = img_BGR[:, :, 0]
    #img_G = img_BGR[:, :, 1]
    #img_R = img_BGR[:, :, 2]
    img_B, img_G, img_R = cv2.split(img_BGR)
    cv2.imshow("B", cv2.merge([img_B, zeros, zeros]))
    cv2.waitKey(0)
    cv2.imshow("G", cv2.merge([zeros, img_G, zeros]))
    cv2.waitKey(0)
    cv2.imshow("R", cv2.merge([zeros, zeros, img_R]))
    cv2.waitKey(0)
    
    img_merge = cv2.merge([img_B, img_G, img_R])
    cv2.imshow("Merge", img_merge)
    cv2.waitKey(0)

    # roi
    # roi = np.ones((100, 100, 3))
    roi = img_BGR[200:300, 200:300]
    #cv2.imshow("roi", roi)
    #cv2.waitKey(0)
    img_BGR[0:100,0:100] = roi
    cv2.imshow("roi", img_BGR)
    cv2.waitKey(0)

    img_GRAY = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
    print(img_GRAY.shape)
    print(img_GRAY.size)
    print(img_GRAY.dtype)
    cv2.imshow("Gray", img_GRAY)
    cv2.waitKey(0)

    cv2.destroyAllWindows()


def fun2(img):
    b,g,r=cv2.split(img) #通道分割
    img_RGB=cv2.merge([r,g,b])#通道组合

    plt.subplot(121),plt.imshow(img),plt.title('img_BGR'), plt.axis('off') #坐标轴关闭
    plt.subplot(122),plt.imshow(img_RGB),plt.title('img_RGB'), plt.axis('off')
    plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    imgpath = 'images/Cats_500X400.jpg'
    images = loadimage(imgpath)
