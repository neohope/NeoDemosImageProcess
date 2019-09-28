#!/usr/bin/python
# -*- coding:utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt


def fft_np(img):
    #快速傅里叶变换算法得到频率分布
    f = np.fft.fft2(img)
    #默认结果中心点位置是在左上角,
    #调用fftshift()函数转移到中间位置
    fshift = np.fft.fftshift(f)       
    #fft结果是复数, 其绝对值结果是振幅
    res = np.log(np.abs(fshift))

    #傅里叶逆变换
    ishift = np.fft.ifftshift(fshift)
    iimg = np.fft.ifft2(ishift)
    iimg = np.abs(iimg)

    #展示结果
    plt.subplot(131), plt.imshow(img, 'gray'), plt.title('Original Image')
    plt.axis('off')
    plt.subplot(132), plt.imshow(res, 'gray'), plt.title('Fourier Image')
    plt.axis('off')
    plt.subplot(133), plt.imshow(iimg, 'gray'), plt.title('Inverse Fourier Image')
    plt.axis('off')
    plt.show()


def fft_cv2(img):
    #傅里叶变换
    dft = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)
    #将频谱低频从左上角移动至中心位置
    dftshift = np.fft.fftshift(dft)
    #频谱图像双通道复数转换为0-255区间
    res1 = 20*np.log(cv2.magnitude(dftshift[:,:,0], dftshift[:,:,1]))

    #傅里叶逆变换
    ishift = np.fft.ifftshift(dftshift)
    iimg = cv2.idft(ishift)
    res2 = cv2.magnitude(iimg[:,:,0], iimg[:,:,1])

    #显示图像
    plt.subplot(131), plt.imshow(img, 'gray'), plt.title('Original Image')
    plt.axis('off')
    plt.subplot(132), plt.imshow(res1, 'gray'), plt.title('Fourier Image')
    plt.axis('off')
    plt.subplot(133), plt.imshow(res2, 'gray'), plt.title('Inverse Fourier Image')
    plt.axis('off')
    plt.show()


# 高通提取图像边缘
def high_pass(img):
    #傅里叶变换
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    #设置高通滤波器
    rows, cols = img.shape
    crow,ccol = int(rows/2), int(cols/2)
    fshift[crow-30:crow+30, ccol-30:ccol+30] = 0

    #傅里叶逆变换
    ishift = np.fft.ifftshift(fshift)
    iimg = np.fft.ifft2(ishift)
    iimg = np.abs(iimg)

    #显示原始图像和高通滤波处理图像
    plt.subplot(121), plt.imshow(img, 'gray'), plt.title('Original Image')
    plt.axis('off')
    plt.subplot(122), plt.imshow(iimg, 'gray'), plt.title('Result Image')
    plt.axis('off')
    plt.show()


# 高通提取图像边缘
def low_pass(img):
    #傅里叶变换
    dft = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)
    fshift = np.fft.fftshift(dft)

    #设置低通滤波器
    rows, cols = img.shape
    crow,ccol = int(rows/2), int(cols/2) #中心位置
    mask = np.zeros((rows, cols, 2), np.uint8)
    mask[crow-30:crow+30, ccol-30:ccol+30] = 1

    #掩膜图像和频谱图像乘积
    f = fshift * mask

    #傅里叶逆变换
    ishift = np.fft.ifftshift(f)
    iimg = cv2.idft(ishift)
    res = cv2.magnitude(iimg[:,:,0], iimg[:,:,1])

    #显示原始图像和低通滤波处理图像
    plt.subplot(121), plt.imshow(img, 'gray'), plt.title('Original Image')
    plt.axis('off')
    plt.subplot(122), plt.imshow(res, 'gray'), plt.title('Result Image')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    srcImage = cv2.imread("images/Forest_500X280.jpg")
    grayImage = cv2.cvtColor(srcImage, cv2.COLOR_BGR2GRAY)
    #fft_np(grayImage)
    #fft_cv2(grayImage)
    high_pass(grayImage)
    low_pass(grayImage)
