# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 01:29:16 2020

@author: Abhishek Ranjan Singh
"""
import cv2
import numpy as np


def hist_equal(img,L):
    '''takes image and an output space [0,L] as an input and gives an equalized image(in float) as output'''
    if len(img.shape) != 2:
        R, G, B = cv2.split(img)
        
        output1_R = cv2.equalizeHist(R)
        output1_G = cv2.equalizeHist(G)
        output1_B = cv2.equalizeHist(B)
        
        equ = cv2.merge((output1_R, output1_G, output1_B))
        return equ
    
    equ = cv2.equalizeHist(img)
    res = np.hstack((img,equ)) #stacking images side-by-side

    return res

def feature_extraction(img = None, Shape = (20,20), flag = 0):
    '''Takes an image as an input and gives a 400x1 vector as output'''
    if type(img) != type(None):
        if (img.shape[0],img.shape[1]) != Shape: #Resizing the image to match the length of feature vector
            img = cv2.resize(img,Shape,interpolation = cv2.INTER_AREA)
        #Performing histogram equalization and also transfoming the image space to [0,1] as computational load will be reduced
        #
        if flag != 0:
            img = hist_equal(img,1)
            
        if len(img.shape) == 2:
            return img.flatten()
        else:
            output = np.zeros((400,1,3))
            output[:,:,0] = np.reshape(img[:,:,0].flatten(),(400,1))
            output[:,:,1] = np.reshape(img[:,:,1].flatten(),(400,1))
            output[:,:,2] = np.reshape(img[:,:,2].flatten(),(400,1))
            return np.reshape(output.flatten(),(1200,1))
    return None
        
    

def main():
    lena = cv2.imread('000097.jpg') 
    
    lenagray = cv2.cvtColor(lena,cv2.COLOR_BGR2GRAY)
    x = feature_extraction(lena)
    #check whether the feature extraction works or not
    print(x.shape)
    print(type(x))
    
    cv2.imshow('Output',hist_equal(lena,255).astype('uint8'))
    cv2.waitKey(0)
    print('HelloWorld')

if __name__ == '__main__':
    main()
    