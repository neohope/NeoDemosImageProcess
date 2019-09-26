#!/usr/bin/python
# -*- coding:utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt


#最大灰度处理
def gray_max(img):
    rows,cols,chanels = img.shape
    grayimg = np.zeros((rows, cols, 3), np.uint8)

    for i in range(rows):
        for j in range(cols):
            #获取图像RGB最大值
            gray = max(img[i,j][0], img[i,j][1], img[i,j][2])
            grayimg[i,j] = np.uint8(gray)
    
    cv2.imshow("src", img)
    cv2.imshow("gray", grayimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#平均灰度处理
def gray_avg(img):
    rows,cols,chanels = img.shape
    grayimg = np.zeros((rows, cols, 3), np.uint8)

    for i in range(rows):
        for j in range(cols):
            #灰度值为RGB三个分量的平均值
            gray = (int(img[i,j][0]) + int(img[i,j][1]) + int(img[i,j][2]))  /  3
            grayimg[i,j] = np.uint8(gray)
    
    cv2.imshow("src", img)
    cv2.imshow("gray", grayimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#加权平均灰度处理
def gray_weight_avg(img):
    rows,cols,chanels = img.shape
    grayimg = np.zeros((rows, cols, 3), np.uint8)

    for i in range(rows):
        for j in range(cols):
            #灰度加权平均法
            gray = 0.30 * img[i,j][0] + 0.59 * img[i,j][1] + 0.11 * img[i,j][2]
            grayimg[i,j] = np.uint8(gray)
    
    cv2.imshow("src", img)
    cv2.imshow("gray", grayimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def gray_process(img):
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rows, cols = grayImage.shape
    
    #"灰度上移"
    blank1 = np.zeros([rows, cols], grayImage.dtype)
    cv2.convertScaleAbs(grayImage,blank1,1,50)

    #对比度增强
    blank2 = np.zeros([rows, cols], grayImage.dtype)
    cv2.convertScaleAbs(grayImage,blank2,1.5,0)

    #对比度减弱
    blank3 = np.zeros([rows, cols], grayImage.dtype)
    cv2.convertScaleAbs(grayImage,blank3,0.5,0)

    #反色
    blank4 = np.zeros([rows, cols], grayImage.dtype)
    blank4[:,:] = 255
    blank4 = cv2.subtract(blank4, grayImage)

    images = [grayImage, blank1, blank2, blank3, blank4]
    titles = ['grayImage', 'blank1', 'blank2', 'blank3', 'blank4']
    for i in range(5):
        plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == '__main__':
    img = cv2.imread("images/Forest_500X280.jpg")
    #gray_max(img)
    #gray_avg(img)
    #gray_weight_avg(img)
    gray_process(img)
