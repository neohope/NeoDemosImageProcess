#!/usr/bin/python
# -*- coding:utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt


def edge_detection(img):
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

    #Scharr算子
    x = cv2.Scharr(grayImage, cv2.CV_32F, 1, 0) #X方向
    y = cv2.Scharr(grayImage, cv2.CV_32F, 0, 1) #Y方向
    absX = cv2.convertScaleAbs(x)       
    absY = cv2.convertScaleAbs(y)
    Scharr = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    
    # Canny算子
    Canny = cv2.Canny(gaussianBlur, 50, 150)

    #LOG算子
    dst = cv2.Laplacian(gaussianBlur, cv2.CV_16S, ksize = 3) #再通过拉普拉斯算子做边缘检测
    LOG = cv2.convertScaleAbs(dst)
    
    
    images = [img_RGB, binary, Roberts, Prewitt, Sobel, Laplacian, Scharr, Canny, LOG]
    titles = ['原始图像', '二值图', 'Roberts算子', 'Prewitt算子', 'Sobel算子', 'Laplacian算子', 'Scharr算子', 'Canny算子', 'LOG算子']
    plt.rcParams['font.sans-serif'] = ['SimHei']
    for i in range(9):
        plt.subplot(3, 3, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()


if __name__ == '__main__':
    img = cv2.imread("images/Forest_500X280.jpg")
    edge_detection(img)
