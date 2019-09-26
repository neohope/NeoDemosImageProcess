#!/usr/bin/python
# -*- coding:utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt


#鼠标事件
def draw(event, x, y, flags, param):
    global en
    #鼠标左键按下开启en值
    if event==cv2.EVENT_LBUTTONDOWN:
        en = True
    #鼠标左键按下并且移动
    elif event==cv2.EVENT_MOUSEMOVE and flags==cv2.EVENT_LBUTTONDOWN:
            #调用函数打马赛克
            if en:
                drawMask(y,x)
            #鼠标左键弹起结束操作
            elif event==cv2.EVENT_LBUTTONUP:
                en = False
          

#图像局部采样操作         
def drawMask(x, y, size=10):
    #size*size采样处理
    m = x // size * size  
    n = y // size * size
    print(str(m)+','+str(n))
    #10*10区域设置为同一像素值
    for i in range(size):
        for j in range(size):
            img[m+i][n+j] = img[m][n]


if __name__ == '__main__':
    img = cv2.imread("images/Forest_500X280.jpg")

    #设置鼠标左键开启
    en = False

    #打开对话框
    cv2.namedWindow('image')

    #调用draw函数设置鼠标操作
    cv2.setMouseCallback('image', draw)

    #循环处理
    while(1):
        cv2.imshow('image', img)
        
        #按ESC键退出
        if cv2.waitKey(10)&0xFF==27:
            break
        #按s键保存图片
        elif cv2.waitKey(10)&0xFF==115:
            cv2.imwrite('out/sava.png', img)

    #退出窗口
    cv2.destroyAllWindows()
