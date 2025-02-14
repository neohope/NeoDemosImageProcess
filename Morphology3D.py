#!/usr/bin/python
# -*- coding:utf-8 -*-

import cv2
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def x1(imgsrc):
    imgd = np.array(cv2.cvtColor(imgsrc,cv2.COLOR_BGR2GRAY)) 

    #准备数据
    sp = imgsrc.shape
    h = int(sp[0])        #图像高度(rows)
    w = int(sp[1])       #图像宽度(colums) of image

    #绘图初始处理
    fig = plt.figure(figsize=(16,12))
    ax = fig.gca(projection="3d")

    x = np.arange(0, w, 1)
    y = np.arange(0, h, 1)
    x, y = np.meshgrid(x,y)
    z = imgd
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm)  

    #自定义z轴
    ax.set_zlim(-10, 255)
    ax.zaxis.set_major_locator(LinearLocator(10))   #设置z轴网格线的疏密
    #将z的value字符串转为float并保留2位小数
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f')) 

    # 设置坐标轴的label和标题
    ax.set_xlabel('x', size=15)
    ax.set_ylabel('y', size=15)
    ax.set_zlabel('z', size=15)
    ax.set_title("surface plot", weight='bold', size=20)

    #添加右侧的色卡条
    fig.colorbar(surf, shrink=0.6, aspect=8)  
    plt.show()


def x2(imgsrc):
    #图像黑帽运算
    kernel = np.ones((10,10), np.uint8)
    result = cv2.morphologyEx(cv2.cvtColor(imgsrc,cv2.COLOR_BGR2GRAY), cv2.MORPH_BLACKHAT, kernel)

    #image类转numpy
    imgd = np.array(result)     

    #准备数据
    sp = result.shape
    h = int(sp[0])        #图像高度(rows)
    w = int(sp[1])       #图像宽度(colums) of image

    #绘图初始处理
    fig = plt.figure(figsize=(8,6))
    ax = fig.gca(projection="3d")

    x = np.arange(0, w, 1)
    y = np.arange(0, h, 1)
    x, y = np.meshgrid(x,y)
    z = imgd
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm)  

    #自定义z轴
    ax.set_zlim(-10, 255)
    ax.zaxis.set_major_locator(LinearLocator(10))   #设置z轴网格线的疏密
    #将z的value字符串转为float并保留2位小数
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f')) 

    # 设置坐标轴的label和标题
    ax.set_xlabel('x', size=15)
    ax.set_ylabel('y', size=15)
    ax.set_zlabel('z', size=15)
    ax.set_title("surface plot", weight='bold', size=20)

    #添加右侧的色卡条
    fig.colorbar(surf, shrink=0.6, aspect=8)  
    plt.show()


if __name__ == '__main__':
    img = cv2.imread("images/Forest_500X280.jpg")
    x1(img)
    x2(img)
