#!/usr/bin/python
# -*- coding:utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt


def adjust(img, height, width):
    #获取图像大小
    rows, cols = img.shape[:2]

    #将源图像高斯模糊
    img_gaus = cv2.GaussianBlur(img, (3,3), 0)
    
    #进行灰度化处理
    gray = cv2.cvtColor(img_gaus,cv2.COLOR_BGR2GRAY)

    #边缘检测（检测出图像的边缘信息）
    edges = cv2.Canny(gray,40,250,apertureSize = 3)
    #cv2.imwrite("out/canny.jpg", edges)

    kernel = np.ones((3,3), np.uint8)
    expansion = cv2.dilate(edges, kernel, iterations=1)

    #通过霍夫变换得到A4纸边缘
    lines = cv2.HoughLinesP(expansion,1,np.pi/180,50,minLineLength=60,maxLineGap=10)

    #下面输出的四个点分别为四个顶点
    for x1,y1,x2,y2 in lines[0]:
        print(x1,y1),(x2,y2)
    for x1,y1,x2,y2 in lines[1]:
        print(x1,y1),(x2,y2)

    #绘制边缘
    for x in range(0,3,2):
        for x1,y1,x2,y2 in lines[x]:
            cv2.line(img, (x1,y1), (x2,y2), (0,255,0), 5)

    x1,y1,x2,y2 = lines[0][0]
    x3,y3,x4,y4 = lines[2][0]

    #根据四个顶点设置图像透视变换矩阵
    pos1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    pos2 = np.float32([[0, 0], [0, height], [width, 0], [width, height]])
    M = cv2.getPerspectiveTransform(pos1, pos2)

    #图像透视变换
    result = cv2.warpPerspective(img, M, (width, height ))

    #显示图像
    images = [img, img_gaus, gray, edges,  result]
    titles = ['img', 'img_gaus', 'gray', 'edges', 'result']
    for i in range(5):
        plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()



def adjustPhone(img, height, width):
    #获取图像大小
    rows, cols = img.shape[:2]

    #将源图像高斯模糊
    img_gaus = cv2.GaussianBlur(img, (3,3), 0)
    
    #进行灰度化处理
    gray = cv2.cvtColor(img_gaus,cv2.COLOR_BGR2GRAY)

    #边缘检测（检测出图像的边缘信息）
    edges = cv2.Canny(gray,40,250,apertureSize = 3)
    #cv2.imwrite("out/canny.jpg", edges)

    kernel = np.ones((3,3), np.uint8)
    expansion = cv2.dilate(edges, kernel, iterations=2)

    #通过霍夫变换得到A4纸边缘
    lines = cv2.HoughLinesP(expansion,1,np.pi/180,50,minLineLength=60,maxLineGap=10)

    #下面输出的四个点分别为四个顶点
    for x1,y1,x2,y2 in lines[0]:
        print(x1,y1),(x2,y2)
    for x1,y1,x2,y2 in lines[1]:
        print(x1,y1),(x2,y2)

    #绘制边缘
    for x in range(0,2):
        for x1,y1,x2,y2 in lines[x]:
            cv2.line(img, (x1,y1), (x2,y2), (0,255,0), 5)

    
    print(lines[0][0])

    #根据四个顶点设置图像透视变换矩阵
    x1,y1,x2,y2 = lines[0][0]
    x3,y3,x4,y4 = lines[1][0]
    pos1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    pos2 = np.float32([[0, 0], [0, height], [width, 0], [width, height]])
    M = cv2.getPerspectiveTransform(pos1, pos2)

    #图像透视变换
    result = cv2.warpPerspective(img, M, (width, height))

    #显示图像
    images = [img, img_gaus, gray, edges, expansion,  result]
    titles = ['img', 'img_gaus', 'gray', 'edges', 'expansion', 'result']
    for i in range(6):
        plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == '__main__':
    img = cv2.imread("images/Paper_297X511.png")
    adjust(img, 272, 190)

    img = cv2.imread("images/Phone1_400X533.jpg")
    adjustPhone(img, 250, 130)
    
    # need adjust
    #img = cv2.imread("images/Phone_400X533.jpg")
    #adjustPhone(img, 250, 130)
