#!/usr/bin/python
# -*- coding:utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt

# white vessel
def subtract(img01, img11, alpha, beta):
    subtract1 = cv2.subtract(img01, img11)
    #cv2.imshow("01-11", subtract)
    #cv2.waitKey(0)

    rows, cols, chn = subtract1.shape
    blank = np.zeros([rows, cols, chn], subtract1.dtype)
    cv2.convertScaleAbs(subtract1,blank,alpha,beta)
    cv2.imshow("alpha="+str(alpha)+", beta="+str(beta), blank)
    cv2.waitKey(0)


# black vessel
def subtract1(img01, img11, alpha, beta):
    subtract1 = cv2.subtract(img01, img11)
    #cv2.imshow("01-11", subtract)
    #cv2.waitKey(0)
    rows, cols, chn = subtract1.shape
    blank = np.zeros([rows, cols, chn], subtract1.dtype)
    blank[:,:,:] = 255
    subtract2 = cv2.subtract(blank, subtract1)
    #cv2.imshow("alpha="+str(alpha)+", beta="+str(beta), subtract2)
    #cv2.waitKey(0)

    rows, cols, chn = subtract1.shape
    blank = np.zeros([rows, cols, chn], subtract2.dtype)
    cv2.convertScaleAbs(subtract2,blank,1,-128)
    #cv2.imshow("alpha="+str(alpha)+", beta="+str(beta), blank)
    #cv2.waitKey(0)

    blank2 = np.zeros([rows, cols, chn], subtract2.dtype)
    cv2.convertScaleAbs(blank,blank2,alpha,beta)
    cv2.imshow("alpha="+str(alpha)+", beta="+str(beta), blank2)
    cv2.waitKey(0)


# bad, why?
def subtract2(img01, img11, alpha, beta):
    subtract1 = cv2.subtract(img11, img01)
    #cv2.imshow("01-11", subtract)
    #cv2.waitKey(0)

    #关键是这个地方，调整对比度效果很差
    rows, cols, chn = subtract1.shape
    blank = np.zeros([rows, cols, chn], subtract1.dtype)
    cv2.convertScaleAbs(subtract1,blank,4,30)
    cv2.imshow("alpha="+str(alpha)+", beta="+str(beta), blank)
    cv2.waitKey(0)

    blank2 = np.zeros([rows, cols, chn], subtract1.dtype)
    cv2.convertScaleAbs(blank,blank2,5,-400)
    cv2.imshow("alpha="+str(alpha)+", beta="+str(beta), blank2)
    cv2.waitKey(0)


if __name__ == '__main__':
    img01 = cv2.imread("dsa/XA.f1.png")
    img11 = cv2.imread("dsa/XA.f11.png")

    subtract(img01,img11,5,64)
    subtract1(img01,img11,5,-500)
    #subtract2(img01,img11, 1, 0)

