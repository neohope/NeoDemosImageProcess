#!/usr/bin/python
# -*- coding:utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt


# 线性变换，图像反转
def liner_transform(img):
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    height = grayImage.shape[0]
    width = grayImage.shape[1]
    result = np.zeros((height, width), np.uint8)
    
    #图像灰度反色变换 s=255-r
    for i in range(height):
        for j in range(width):
            gray = 255 - grayImage[i,j]
            result[i,j] = np.uint8(gray)
 
    cv2.imshow("Gray Image", grayImage)
    cv2.imshow("liner_transform", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#绘制对数曲线
def log_plot(c):
    x = np.arange(0, 256, 0.01)
    y = c * np.log(1 + x)
    plt.plot(x, y, 'r', linewidth=1)
    plt.rcParams['font.sans-serif']=['SimHei'] #正常显示中文标签
    plt.title(u'对数变换函数')
    plt.xlim(0, 255), plt.ylim(0, 255)
    plt.show()


#对数计算
def log(c, img):
    output = c * np.log(1.0 + img)
    output = np.uint8(output + 0.5)
    return output
 

#对数变换
def log_transform(img):
    #绘制对数变换曲线
    log_plot(42)
    
    #图像灰度对数变换
    output = log(42, img)
    
    cv2.imshow('Input', img)
    cv2.imshow('log_transform', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#绘制gamma曲线
def gamma_plot(c, v):
    x = np.arange(0, 256, 0.01)
    y = c*x**v
    plt.plot(x, y, 'r', linewidth=1)
    plt.rcParams['font.sans-serif']=['SimHei'] #正常显示中文标签
    plt.title(u'伽马变换函数')
    plt.xlim([0, 255]), plt.ylim([0, 255])
    plt.show()


#伽玛计算
def gamma(img, c, v):
    lut = np.zeros(256, dtype=np.float32)
    for i in range(256):
        lut[i] = c * i ** v
    output_img = cv2.LUT(img, lut) #像素灰度值的映射
    output_img = np.uint8(output_img+0.5)
    return output_img


#伽玛变换
def gamma_transform(img):
    #绘制伽玛变换曲线
    gamma_plot(0.00000005, 4.0)
    
    #图像灰度伽玛变换
    output = gamma(img, 0.00000005, 4.0)
    
    cv2.imshow('Imput', img)
    cv2.imshow('gamma_transform', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    img = cv2.imread("images/Forest_500X280.jpg")
    liner_transform(img)
    log_transform(img)
    gamma_transform(img)
