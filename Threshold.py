#!/usr/bin/python
# -*- coding:utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt


#阈值化处理
def threshold(img): 
    img_Gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    # 二进制阈值化，大于某个值为255，小于某个值为0
    ret,thresh1=cv2.threshold(img_Gray,127,255,cv2.THRESH_BINARY)
    # 反二进制阈值化，大于某个值为0，小于某个值为255
    ret,thresh2=cv2.threshold(img_Gray,127,255,cv2.THRESH_BINARY_INV)
    # 截断阈值化，大于某个值为该值(还是255?)，小于某个值不变
    ret,thresh3=cv2.threshold(img_Gray,127,255,cv2.THRESH_TRUNC)
    # 反阈值化为0，大于某个值为0，小于某个值不变
    ret,thresh4=cv2.threshold(img_Gray,127,255,cv2.THRESH_TOZERO)
    # 阈值化为0，大于某个值不变，小于某个值为0
    ret,thresh5=cv2.threshold(img_Gray,127,255,cv2.THRESH_TOZERO_INV)
    
    titles = ['Gray Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
    images = [img_Gray, thresh1, thresh2, thresh3, thresh4, thresh5]
    for i in range(6):
        plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()


if __name__ == '__main__':
    img = cv2.imread("images/Forest_500X280.jpg")
    threshold(img)
