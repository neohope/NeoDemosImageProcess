#!/usr/bin/python
# -*- coding:utf-8 -*-

import cv2
import math
import numpy as np
import matplotlib.pyplot as plt


#毛玻璃
def ground_glass(img):
    #新建目标图像
    dst = np.zeros_like(img)

    #获取图像行和列
    rows, cols = img.shape[:2]

    #定义偏移量和随机数
    offsets = 5
    random_num = 0

    #毛玻璃效果: 像素点邻域内随机像素点的颜色替代当前像素点的颜色
    for y in range(rows - offsets):
        for x in range(cols - offsets):
            random_num = np.random.randint(0,offsets)
            dst[y,x] = img[y + random_num,x + random_num]

    #显示图像
    cv2.imshow('src',img)
    cv2.imshow('ground_glass',dst)

    cv2.waitKey()
    cv2.destroyAllWindows()


# 浮雕
def relief(img):
    #获取图像的高度和宽度
    height, width = img.shape[:2]

    #图像灰度处理
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #创建目标图像
    dstImg = np.zeros((height,width,1),np.uint8)

    #浮雕特效算法：newPixel = grayCurrentPixel - grayNextPixel + 150
    for i in range(0,height):
        for j in range(0,width-1):
            grayCurrentPixel = int(gray[i,j])
            grayNextPixel = int(gray[i,j+1])
            newPixel = grayCurrentPixel - grayNextPixel + 150
            if newPixel > 255:
                newPixel = 255
            if newPixel < 0:
                newPixel = 0
            dstImg[i,j] = newPixel
            
    #显示图像
    cv2.imshow('src', img)
    cv2.imshow('relief',dstImg)

    #等待显示
    cv2.waitKey()
    cv2.destroyAllWindows()


#油漆
def painting(img):
    #图像灰度处理
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #自定义卷积核
    kernel = np.array([[-1,-1,-1],[-1,10,-1],[-1,-1,-1]])

    #图像浮雕效果
    output = cv2.filter2D(gray, -1, kernel)

    #显示图像
    cv2.imshow('src', img)
    cv2.imshow('painting',output)

    #等待显示
    cv2.waitKey()
    cv2.destroyAllWindows()


#素描
def sketch(img):
    #图像灰度处理
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #高斯滤波降噪
    gaussian = cv2.GaussianBlur(gray, (5,5), 0)
    
    #Canny算子
    canny = cv2.Canny(gaussian, 50, 150)

    #阈值化处理
    ret, result = cv2.threshold(canny, 100, 255, cv2.THRESH_BINARY_INV)

    #显示图像
    cv2.imshow('src', img)
    cv2.imshow('sketch', result)
    cv2.waitKey()
    cv2.destroyAllWindows()


# 怀旧
def reminiscent(img):
    #获取图像行和列
    rows, cols = img.shape[:2]

    #新建目标图像
    dst = np.zeros((rows, cols, 3), dtype="uint8")

    #图像怀旧特效
    for i in range(rows):
        for j in range(cols):
            B = 0.272*img[i,j][2] + 0.534*img[i,j][1] + 0.131*img[i,j][0]
            G = 0.349*img[i,j][2] + 0.686*img[i,j][1] + 0.168*img[i,j][0]
            R = 0.393*img[i,j][2] + 0.769*img[i,j][1] + 0.189*img[i,j][0]
            if B>255:
                B = 255
            if G>255:
                G = 255
            if R>255:
                R = 255
            dst[i,j] = np.uint8((B, G, R))
            
    #显示图像
    cv2.imshow('src', img)
    cv2.imshow('reminiscent', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()


#光照
def brighten(img):
    #获取图像行和列
    rows, cols = img.shape[:2]

    #设置中心点
    centerX = rows / 2
    centerY = cols / 2
    radius = min(centerX, centerY)

    #设置光照强度
    strength = 200

    #新建目标图像
    dst = np.zeros((rows, cols, 3), dtype="uint8")

    #图像光照特效
    for i in range(rows):
        for j in range(cols):
            #计算当前点到光照中心距离(平面坐标系中两点之间的距离)
            distance = math.pow((centerY-j), 2) + math.pow((centerX-i), 2)
            #获取原始图像
            B =  img[i,j][0]
            G =  img[i,j][1]
            R = img[i,j][2]
            if (distance < radius * radius):
                #按照距离大小计算增强的光照值
                result = (int)(strength*( 1.0 - math.sqrt(distance) / radius ))
                B = img[i,j][0] + result
                G = img[i,j][1] + result
                R = img[i,j][2] + result
                #判断边界 防止越界
                B = min(255, max(0, B))
                G = min(255, max(0, G))
                R = min(255, max(0, R))
                dst[i,j] = np.uint8((B, G, R))
            else:
                dst[i,j] = np.uint8((B, G, R))
            
    #显示图像
    cv2.imshow('src', img)
    cv2.imshow('brighten', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()


#流年
def fleeting_time(img):
    #获取图像行和列
    rows, cols = img.shape[:2]

    #新建目标图像
    dst = np.zeros((rows, cols, 3), dtype="uint8")

    #图像流年特效
    for i in range(rows):
        for j in range(cols):
            #B通道的数值开平方乘以参数12
            B = math.sqrt(img[i,j][0]) * 12
            G =  img[i,j][1]
            R =  img[i,j][2]
            if B>255:
                B = 255
            dst[i,j] = np.uint8((B, G, R))
            
    #显示图像
    cv2.imshow('src', img)
    cv2.imshow('fleeting_time', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()


#获取滤镜颜色
def getBGR(img, ctable, i, j):
    #获取图像颜色
    b, g, r = img[i][j]

    #计算标准颜色表中颜色的位置坐标
    x = int(g//4 + int(b//32) * 64)
    y = int(r//4 + int((b%32) // 4) * 64)
    
    #返回滤镜颜色表中对应的颜色
    return ctable[x][y]


#滤镜
def color_filter(img):
    #读取原始图像
    ctable = cv2.imread('images/ColorTable.png')

    #获取图像行和列
    rows, cols = img.shape[:2]

    #新建目标图像
    dst = np.zeros((rows, cols, 3), dtype="uint8")

    #循环设置滤镜颜色
    for i in range(rows):
        for j in range(cols):
            dst[i][j] = getBGR(img, ctable, i, j)
            
    #显示图像
    cv2.imshow('src', img)
    cv2.imshow('dst', dst)

    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    srcImage = cv2.imread("images/Forest_500X280.jpg")
    # ground_glass(srcImage)
    # relief(srcImage)
    # painting(srcImage)
    # sketch(srcImage)
    # reminiscent(srcImage)
    # brighten(srcImage)
    # fleeting_time(srcImage)
    color_filter(srcImage)
