#!/usr/bin/python
# -*- coding:utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt


def fun1(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equ = cv2.equalizeHist(img_gray)

    plt.subplot(221),plt.imshow(img_gray, 'gray'),plt.title('img'), plt.xticks([]),plt.yticks([])
    plt.subplot(222),plt.imshow(equ, 'gray'),plt.title('equ'), plt.xticks([]),plt.yticks([])
    plt.subplot(223),plt.hist(img_gray.ravel(),256),plt.title('img_hist')
    plt.subplot(224),plt.hist(equ.ravel(),256),plt.title('equ_hist')
    plt.show()
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def fun2(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    plt.subplot(221),plt.imshow(img),plt.title('img'), plt.axis('off') 
    plt.subplot(222),plt.imshow(img, cmap=plt.cm.gray),plt.title('img_cmap'), plt.axis('off')
    plt.subplot(223),plt.imshow(img_gray),plt.title('img_gray'), plt.axis('off')
    plt.subplot(224),plt.imshow(img_gray, cmap=plt.cm.gray),plt.title('img_gray_cmap'),plt.axis('off')#正确用法
    plt.show()
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def fun3(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equ = cv2.equalizeHist(img_gray)
 
    plt.subplot(221),plt.imshow(img_gray, cmap=plt.cm.gray),plt.title('img_gray'), plt.axis('off') #坐标轴关闭
    plt.subplot(222),plt.imshow(equ, cmap=plt.cm.gray),plt.title('equ'), plt.axis('off') #坐标轴关闭
    plt.subplot(223),plt.hist(img_gray.ravel(),256),plt.title('img_gray_hist')
    plt.subplot(224),plt.hist(equ.ravel(),256),plt.title('equ_hist')
    plt.show()
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    img = cv2.imread("images/Forest_500X280.jpg")
    fun1(img)
    fun2(img)
    fun3(img)
