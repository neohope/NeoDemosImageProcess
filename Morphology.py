#!/usr/bin/python
# -*- coding:utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt


#腐蚀
def erosion(img):
    kernel = np.ones((3,3), np.uint8)
    erosion = cv2.erode(img, kernel, iterations=2)
    cv2.imshow("img", img)
    cv2.imshow("erosion", erosion)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#膨胀
def expansion(img):
    kernel = np.ones((3,3), np.uint8)
    expansion = cv2.dilate(img, kernel, iterations=2)
    cv2.imshow("img", img)
    cv2.imshow("expansion", expansion)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#图像开运算(先腐蚀，后膨胀)
def morphopen(img):
    kernel = np.ones((3,3), np.uint8)
    result = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    cv2.imshow("result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#图像闭运算(先膨胀，后腐蚀)
def morphclose(img):
    kernel = np.ones((3,3), np.uint8)
    result = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#图像梯度运算(膨胀减去腐蚀)
def morphgradient(img):
    kernel = np.ones((3,3), np.uint8)
    result = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    cv2.imshow("result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#图像顶帽运算（原始减去开运算)
def morphtophat(img):
    kernel = np.ones((3,3), np.uint8)
    result = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    cv2.imshow("result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#图像黑帽运算（闭运算减去原始)
def morphblackhat(img):
    kernel = np.ones((3,3), np.uint8)
    result = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
    cv2.imshow("result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    img = cv2.imread("images/Forest_500X280.jpg")
    #erosion(img)
    #expansion(img)
    #morphopen(img)
    #morphclose(img)
    #morphgradient(img)
    morphtophat(img)
    morphblackhat(img)
