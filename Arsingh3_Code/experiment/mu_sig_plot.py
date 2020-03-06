# -*- coding: utf-8 -*-

import cv2
import numpy as np

def plot_mu_sig (mu_1,sig_1,title1,title2,img_type,image_size,ploting):
    # Plot Mean  
    mu = mu_1 / np.max(mu_1)
    if (img_type == 'GRAY'):
        mu_mat = np.reshape(mu,(image_size,image_size))
    else:
        mu_mat = np.reshape(mu,(image_size,image_size,3))
    r = 200.0 / mu_mat.shape[1]
    dim = (200, int(mu_mat.shape[0] * r))
    resized = cv2.resize(mu_mat, dim, interpolation = cv2.INTER_AREA)
    cv2.imshow("resized", resized)
    resized = resized*(255/np.max(resized))
    k = cv2.waitKey(0) & 0xFF
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite(title1+'.png',resized.astype('uint8'))
        cv2.destroyAllWindows()
     
    if (ploting == 'BOTH'):
        #% Plot covariane
        sig = np.zeros(sig_1.shape[0])
        for i in range(sig_1.shape[0]):
            sig[i] = sig_1[i][i]
        sig = sig/np.max(sig)
        if (img_type == 'GRAY'):
            sig_mat = np.reshape((sig),(image_size,image_size))
        else:
            sig_mat = np.reshape((sig),(image_size,image_size,3))
        r = 200.0 / sig_mat.shape[1]
        dim = (200, int(sig_mat.shape[0] * r))
        # perform the actual resizing of the image and show it
        resized = cv2.resize(sig_mat, dim, interpolation = cv2.INTER_AREA)
        cv2.imshow("resized", resized)
        resized = resized*(255/np.max(resized))
        k = cv2.waitKey(0) & 0xFF
        if k == 27:         # wait for ESC key to exit
            cv2.destroyAllWindows()
        elif k == ord('s'): # wait for 's' key to save and exit
            cv2.imwrite(title2 +'.png',resized)
            cv2.destroyAllWindows()
             