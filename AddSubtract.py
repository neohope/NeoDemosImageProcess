#!/usr/bin/python
# -*- coding:utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt

#Numpy：(a+b)%255
#OpenCV：Math.min(a+b,255)
def add(img1,img2):
    result = cv2.add(img1, img2)
    cv2.imshow("add", result)
    cv2.waitKey(0)


def addWeighted(img1, alpha, img2, beta, gamma):
    result  = cv2.addWeighted(img1,alpha,img2,beta,gamma)
    cv2.imshow("addWeighted", result)
    cv2.waitKey(0)


def subtract(img1, img2):
    result = cv2.subtract(img1, img2)
    cv2.imshow("subtract", result)
    cv2.waitKey(0)


if __name__ == '__main__':
    img1 = cv2.imread("images/Forest_500X280.jpg")
    print(img1.shape)
    img2 = cv2.imread("images/Forest1_500X280.jpg")
    print(img2.shape)

    #add(img1, img2)
    #addWeighted(img1, 0.2, img2, 0.8, 0)
    addWeighted(img1, 0.3, img2, 1.0, 0)
    #subtract(img1,img2)
