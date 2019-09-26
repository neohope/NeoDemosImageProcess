#!/usr/bin/python
# -*- coding:utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt


def quantization(img):
    height = img.shape[0]
    width = img.shape[1]

    new_img1 = np.zeros((height, width, 3), np.uint8)
    new_img2 = np.zeros((height, width, 3), np.uint8)
    new_img3 = np.zeros((height, width, 3), np.uint8)

    #图像量化等级为2的量化处理
    for i in range(height):
        for j in range(width):
            for k in range(3): #对应BGR三分量
                if img[i, j][k] < 128:
                    gray = 0
                else:
                    gray = 128
                new_img1[i, j][k] = np.uint8(gray)

    #图像量化等级为4的量化处理
    for i in range(height):
        for j in range(width):
            for k in range(3): #对应BGR三分量
                if img[i, j][k] < 64:
                    gray = 0
                elif img[i, j][k] < 128:
                    gray = 64
                elif img[i, j][k] < 192:
                    gray = 128
                else:
                    gray = 192
                new_img2[i, j][k] = np.uint8(gray)

    #图像量化等级为8的量化处理
    for i in range(height):
        for j in range(width):
            for k in range(3): #对应BGR三分量
                if img[i, j][k] < 32:
                    gray = 0
                elif img[i, j][k] < 64:
                    gray = 32
                elif img[i, j][k] < 96:
                    gray = 64
                elif img[i, j][k] < 128:
                    gray = 96
                elif img[i, j][k] < 160:
                    gray = 128
                elif img[i, j][k] < 192:
                    gray = 160
                elif img[i, j][k] < 224:
                    gray = 192
                else:
                    gray = 224
                new_img3[i, j][k] = np.uint8(gray)

    #用来正常显示中文标签
    plt.rcParams['font.sans-serif']=['SimHei']

    #显示图像
    titles = [u'(a) 原始图像', u'(b) 量化-L2', u'(c) 量化-L4', u'(d) 量化-L8']  
    images = [img, new_img1, new_img2, new_img3]  
    for i in range(4):  
        plt.subplot(2,2,i+1), plt.imshow(images[i], 'gray'), 
        plt.title(titles[i])  
        plt.xticks([]),plt.yticks([])  
    plt.show()


def kmean_quantization(img):
    #图像二维像素转换为一维
    data = img.reshape((-1,3))
    data = np.float32(data)

    #定义中心 (type,max_iter,epsilon)
    criteria = (cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    #设置标签
    flags = cv2.KMEANS_RANDOM_CENTERS

    #K-Means聚类 聚集成4类
    compactness, labels, centers = cv2.kmeans(data, 4, None, criteria, 10, flags)


    #图像转换回uint8二维类型
    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    dst = res.reshape((img.shape))

    #图像转换为RGB显示
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)


    #用来正常显示中文标签
    plt.rcParams['font.sans-serif']=['SimHei']

    #显示图像
    titles = [u'原始图像', u'聚类量化 K=4']  
    images = [img, dst]  
    for i in range(2):  
        plt.subplot(1,2,i+1), plt.imshow(images[i], 'gray'), 
        plt.title(titles[i])  
        plt.xticks([]),plt.yticks([])  
    plt.show()


if __name__ == '__main__':
    img = cv2.imread("images/Forest_500X280.jpg")
    #grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    quantization(img)
    kmean_quantization(img)
