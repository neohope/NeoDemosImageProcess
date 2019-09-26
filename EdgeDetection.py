#!/usr/bin/python
# -*- coding:utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt


def edge(img):
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #转成RGB 方便后面显示
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #高斯滤波
    gaussianBlur = cv2.GaussianBlur(grayImage, (3,3), 0)
    #阈值处理
    ret, binary = cv2.threshold(gaussianBlur, 127, 255, cv2.THRESH_BINARY)
    #Roberts算子
    kernelx = np.array([[-1,0],[0,1]], dtype=int)
    kernely = np.array([[0,-1],[1,0]], dtype=int)
    x = cv2.filter2D(binary, cv2.CV_16S, kernelx)
    y = cv2.filter2D(binary, cv2.CV_16S, kernely)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Roberts = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    
    #Prewitt算子
    kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]], dtype=int)
    kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]], dtype=int)
    x = cv2.filter2D(binary, cv2.CV_16S, kernelx)
    y = cv2.filter2D(binary, cv2.CV_16S, kernely)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Prewitt = cv2.addWeighted(absX,0.5,absY,0.5,0)
    
    #Sobel算子
    x = cv2.Sobel(binary, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(binary, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    
    #Laplacian算子
    dst = cv2.Laplacian(binary, cv2.CV_16S, ksize = 3)
    Laplacian = cv2.convertScaleAbs(dst)
    
    # Canny算子
    Canny = cv2.Canny(gaussianBlur, 50, 150)
    
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.subplot(241),plt.imshow(img_RGB),plt.title('原始图像'), plt.axis('off') #坐标轴关闭
    plt.subplot(242),plt.imshow(binary, cmap=plt.cm.gray ),plt.title('二值图'), plt.axis('off')
    plt.subplot(243),plt.imshow(Roberts, cmap=plt.cm.gray ),plt.title('Roberts算子'), plt.axis('off')
    plt.subplot(244),plt.imshow(Prewitt, cmap=plt.cm.gray ),plt.title('Prewitt算子'), plt.axis('off')
    plt.subplot(245),plt.imshow(Sobel, cmap=plt.cm.gray ),plt.title('Sobel算子'), plt.axis('off')
    plt.subplot(246),plt.imshow(Laplacian, cmap=plt.cm.gray ),plt.title('Laplacian算子'), plt.axis('off')
    plt.subplot(247),plt.imshow(Canny, cmap=plt.cm.gray ),plt.title('Canny算子'), plt.axis('off')
    plt.show()


if __name__ == '__main__':
    img = cv2.imread("images/Forest_500X280.jpg")
    edge(img)
