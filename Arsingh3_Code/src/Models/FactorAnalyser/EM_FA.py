
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 18:14:01 2020

@author: abhis
"""
import numpy as np

def EM(data,phi0,mu0,sig0,iterations = 10):
    '''Data is a numpy arra
    Input Parameters: 
    data = 1200,1000 np.ndarray
    phi0 = 1200,number of factors
    mu0 = 1200,
    sig0 = 1200,
    I ->1000
    D -> 1200
    K-> nbFactors
    
        '''
        
    phi = phi0
    mu = mu0
    sig = sig0
    
    (D,I) = np.shape(data) #D = 1200, I = 1000
#    print('Shape of Data {}'.format(data.shape))
    K = phi.shape[1] #K =3
     
    for iterate in range(iterations):
        
        
        #E-Step:
        x_minus_mu = np.subtract(data,mu.reshape(mu.shape[0],1)) # xdim = 1200,1000
        
        inv_sig = np.diag(1 / sig) #1000,1000
#        print('shape of iverse sig'.format(inv_sig.shape))
        phi_transpose_times_sig_inv = np.dot(np.transpose(phi),inv_sig)
#        print(phi_transpose_times_sig_inv.shape)
        temp = np.linalg.inv(np.dot(phi_transpose_times_sig_inv,phi) + np.identity(K))
        #temp = (3,3)
        E_hi = np.dot(np.dot(temp,phi_transpose_times_sig_inv),x_minus_mu)
        #E,hi = 3,1000
        E_hi_hitr = [[]]*I
#        print('dim of E_hi_tr: {}'.format(len(E_hi_hitr)))
        for i in range (I):
            e = E_hi[:,i]
            E_hi_hitr[i] = temp + np.dot(e,np.transpose(e))
        
        #M-step
        #Update Phi
        phi_1 = np.zeros([D,K])
        for i in range(I):
            sub1 = x_minus_mu[:,i].reshape(x_minus_mu[:,i].shape[0],1)
            sub2 = np.transpose(np.reshape(E_hi[:,i],(-1,1)))
            phi_1 = phi_1 + np.dot(sub1,sub2)
        
        phi_2 = np.zeros([K,K])
        for i in range(I):
            phi_2 = phi_2 + E_hi_hitr[i]
        phi_2 = np.linalg.inv(phi_2)
        phi = np.dot(phi_1,phi_2)
        
        #Update Sig,
        sig_diag = np.zeros([D,1])
        for i in range(I):
            xm = np.transpose(x_minus_mu[:,i])
            sig_1 = xm * xm
            sig_2 = np.dot(phi,E_hi[:,i]) * xm;
            sig_diag = sig_diag + sig_1 - sig_2
        sig = sig_diag / I
        sig = np.diag(sig)
        
        
    return phi,mu,sig
        