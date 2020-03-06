# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 02:29:27 2020

@author: Abhihsek Ranjan Singh
"""

import cv2
import numpy as np
#import os
import glob
import random
import pandas as pd

def data_extraction(faceflag = 0):
    labels = glob.glob('.\FDDB-folds\*.txt')

    l_contents = []
    for i in range(len(labels)):
                 # list of lines of textfile
                 with open(labels[i], 'r') as f:
                     f_contents = f.readlines()
                     for a in f_contents:
                         a = a.rstrip("\n")
                         l_contents.append(a)
    ix = 0
    cf = 0
    cnf = 0
    for i in l_contents:
        try:
            i = int(i)
            if i==1:
                #path
                path_of_image = 'originalPics/' + l_contents[ix-1]

                temp = l_contents[ix+1].split(" ")
                maj_ax = round(float(temp[0]))
                min_ax = round(float(temp[1]))
                angle = round(float(temp[2]))
                cent_x = round(float(temp[3]))
                cent_y = round(float(temp[4]))
                
    
                img = cv2.imread((path_of_image + '.jpg'))
    
                '''Extracting Face from a given Image in rectangular frame'''
                (w,h) = img.shape[:2]
                w = 2*w
                h = 2*h
                M = cv2.getRotationMatrix2D((cent_x,cent_y), angle, 1)
                rotated = cv2.warpAffine(img, M, (w, h))
                face_img = rotated[cent_y - maj_ax:cent_y + maj_ax, cent_x - min_ax:cent_x + min_ax]
                print(face_img.shape)
                cv2.imwrite('annotated_data/face/image{}.jpg'.format(cf), face_img)
                
                '''Extracting non-face from the same image'''
                if faceflag == 1:    
                    pix1 = np.random.randint(low = 0,high = cent_y-maj_ax)
                    pix2 = np.random.randint(low = cent_y+ maj_ax,high = img.shape[1])
                    startpix = random.choice([pix1,pix2])
                    if (startpix + int(maj_ax)) < img.shape[0] and (startpix + int(min_ax)) < img.shape[1]:
                        non_face_img = img[startpix:startpix + int(maj_ax),startpix:startpix + int(min_ax)]
                        cv2.imwrite('annotated_data/Non_face/image{}.jpg'.format(cnf), non_face_img)
                        cnf = cnf+1
                else:
                    '''SubmissionFace'''
                    print("its happening")
                    nonface = img[0:60,0:60]
                    cv2.imwrite('annotated_data/Non_face/image{}.jpg'.format(cnf), nonface)
                    cnf = cnf+1
                cf = cf+1
        except Exception:
            pass
        ix = ix + 1

def to_CSV():
    faceimages = glob.glob(r'annotated_data//face//*.jpg')
    nonfaceimages = glob.glob(r'annotated_data//Non_face//*.jpg')
    faceimages = {'Image':glob.glob(r'annotated_data//face//*.jpg'),'IsFace':np.ones(len(glob.glob(r'annotated_data//face//*.jpg')))}
    nonfaceimages = {'Image':glob.glob(r'annotated_data//Non_face//*.jpg'),'IsFace':np.zeros(len(glob.glob(r'annotated_data//Non_face//*.jpg')))}
    df1 = pd.DataFrame(faceimages)
    df2 = pd.DataFrame(nonfaceimages)
    df1.to_csv('faceData.csv',index=False)
    df2.to_csv('NonFaceData.csv',index=False)
    

def main():
    #data_extraction(faceflag=0)
    
    to_CSV()
    
    print('Work in Progress')
    
if __name__ == '__main__':
    main()
