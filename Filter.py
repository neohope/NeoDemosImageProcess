#!/usr/bin/python
# -*- coding:utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt


def addSomeNoize(img):
    img_noise=img
    rows, cols, chn = img_noise.shape

    for i in range(5000):
        x = np.random.randint(0, rows)
        y = np.random.randint(0, cols)
        img_noise[x, y, :] = 255
    
    #cv2.imwrite("noise.jpg", img_noise)


#均值滤波
def meanFilter(img):
    source = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img3 = cv2.blur(source, (3, 3))
    img5 = cv2.blur(source, (5, 5))
    img9 = cv2.blur(source, (9, 9))

    titles = ['img', 'img3', 'img5', 'img9']
    images = [source, img3, img5, img9]
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])
    plt.show()



#中值滤波
def medianFilter(img):
    source = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img3 = cv2.medianBlur(source, 3)
    img5 = cv2.medianBlur(source, 5)
    img9 = cv2.medianBlur(source, 9)

    titles = ['img', 'img3', 'img5', 'img9']
    images = [source, img3, img5, img9]
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])
    plt.show()


#方框滤波
def boxFilter(img):
    source = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img3 = cv2.boxFilter(source, -1, (3, 3), normalize=1)
    img5 = cv2.boxFilter(source, -1, (5, 5), normalize=1)
    img9 = cv2.boxFilter(source, -1, (9, 9), normalize=1)

    titles = ['img', 'img3', 'img5', 'img9']
    images = [source, img3, img5, img9]
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])
    plt.show()


#高斯滤波
def gaussianFilter(img):
    source = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img3 = cv2.GaussianBlur(source, (3, 3), 0)
    img5 = cv2.GaussianBlur(source, (5, 5), 0)
    img9 = cv2.GaussianBlur(source, (9, 9), 0)

    titles = ['img', 'img3', 'img5', 'img9']
    images = [source, img3, img5, img9]
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])
    plt.show()


if __name__ == '__main__':
    img = cv2.imread("images/Forest_500X280.jpg")
    addSomeNoize(img)
    #meanFilter(img)
    #medianFilter(img)
    boxFilter(img)
    #gaussianFilter(img)
