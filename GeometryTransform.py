#!/usr/bin/python
# -*- coding:utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 缩放
def zoom(img):
    result1 = cv2.resize(img, (100,56))
    result2 = cv2.resize(img, None, fx=0.1, fy=0.1)
    
    cv2.imshow("img", img)
    cv2.waitKey(0)

    cv2.imshow("result1", result1)
    cv2.waitKey(0)

    cv2.imshow("result2", result2)
    cv2.waitKey(0)


# 旋转
def rotate(img):
    rows, cols, channel = img.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)
    rotated = cv2.warpAffine(img, M, (cols, rows))
    cv2.imshow("rotated", rotated)
    cv2.waitKey(0)


# 翻转
def flip(img):
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img1 = cv2.flip(img_RGB, 0)
    img2 = cv2.flip(img_RGB, 1)
    img3 = cv2.flip(img_RGB, -1)

    titles = ['Source', 'flip0', 'flip1', 'flip-1']
    images = [img_RGB, img1, img2, img3]
    for i in range(4):
        plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()
    
    # 等待显示
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#平移
#[1 0 tx]
#[0 1 ty]
def shift(img):
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rows, cols, chanels = img_RGB.shape

    M = np.float32([[1, 0, 0], [0, 1, -100]])
    img1 = cv2.warpAffine(img_RGB, M, (cols, rows))
    
    M = np.float32([[1, 0, 0], [0, 1, 100]])
    img2 = cv2.warpAffine(img_RGB, M, (cols, rows))
    
    M = np.float32([[1, 0, -100], [0, 1, 0]])
    img3 = cv2.warpAffine(img_RGB, M, (cols, rows))
    
    M = np.float32([[1, 0, 100], [0, 1, 0]])
    img4 = cv2.warpAffine(img_RGB, M, (cols, rows))
    
    titles = ['up', 'down', 'left', 'right']
    images = [img1, img2, img3, img4]
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])
    plt.show()


# 仿射变换/平面变换
def affineTransform(img):
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rows,cols,chanels = img_RGB.shape

    pts1 = np.float32([[0,0],[0,10],[10,0]])
    pts2 = np.float32([[0,0],[1,6],[6,1]])
    M = cv2.getAffineTransform(pts1,pts2)
    img1 = cv2.warpAffine(img_RGB,M,(cols,rows))

    plt.subplot(1,2,1),plt.imshow(img_RGB),plt.title('img')
    plt.subplot(1,2,2),plt.imshow(img1),plt.title('img1')
    plt.show()


# 透视变换/空间变换
def perspectiveTransform(img):
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rows,cols,chanels = img_RGB.shape

    pts1 = np.float32([[0,0],[0,10],[10,0],[10,10]])
    pts2 = np.float32([[0,0],[0,10],[10,0],[10,10]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    img1 = cv2.warpPerspective(img_RGB,M,(cols,rows))

    plt.subplot(1,2,1),plt.imshow(img_RGB),plt.title('img')
    plt.subplot(1,2,2),plt.imshow(img1),plt.title('img1')
    plt.show()


if __name__ == '__main__':
    img = cv2.imread("images/Forest_500X280.jpg")
    #zoom(img)
    #rotate(img)
    #flp(img)
    #shift(img)
    #affineTransform(img)
    perspectiveTransform(img)
