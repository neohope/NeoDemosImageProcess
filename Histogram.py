
#!/usr/bin/python
# -*- coding:utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt


def histogram_plt(img):
    plt.hist(img.ravel(), 256)
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def histogram_cv2(img):
    histb = cv2.calcHist([img], [0], None, [256], [0,255])
    histg = cv2.calcHist([img], [1], None, [256], [0,255])
    histr = cv2.calcHist([img], [2], None, [256], [0,255])

    plt.plot(histb, color='b')
    plt.plot(histg, color='g')
    plt.plot(histr, color='r')
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == '__main__':
    img = cv2.imread("images/Forest_500X280.jpg")
    histogram_plt(img)
    histogram_cv2(img)
