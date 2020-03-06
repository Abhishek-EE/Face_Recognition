# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 18:14:01 2020

@author: abhis
"""
import numpy as np
import src.Distributions.gaussPDF as gaussPDF
import sys
import time
import scipy.special as sc

def EM(data,prior0,mu0,sig0,iterations = 10):
    '''Data is a numpy arra
    Input Parameters: 
    Data is a np array of (1200,1000)
    prior0 is a np array of (1,10)
    mean is np array of (1200,10)
    var is a np array of (1200,1200,10)'''
    realmax = sys.float_info[0]
    realmin = sys.float_info[3]
    loglik_threshold = 1e-10
    prior = prior0
    mu = mu0
    sig = sig0
    loglik_old = realmax
    
    Pij = np.ndarray((data.shape[1],prior.shape[1]))#Dimensions of Pij = (1000,k)
    logPij = np.ndarray((data.shape[1],prior.shape[1]))
    
    for iterate in range(iterations):
        start_main_loop = time.time()
        
        #E-Step:
        
        for  j in range(prior.shape[1]):
            logPij[:,j] = np.nan_to_num((np.log(prior[0,j]+realmin) + gaussPDF.logGaussPdf(data,mu[:,j].reshape(mu.shape[0],1),sig[:,:,j])).reshape((data.shape[1])))
        
        Pij = sc.softmax(logPij,axis = 1)
        
#        total_prob = np.sum(Pij,axis = 1)+realmin
#        total_prob = total_prob.reshape((total_prob.shape[0],1))
#        total_prob = np.broadcast_to(total_prob,Pij.shape)
#        
        print('first two Loops Time: {}'.format(time.time()-start_main_loop))
        
        #M-Step:
        for j in range(prior.shape[1]):
            prior[0,j] = (np.sum(Pij[:,j]))/Pij.shape[0]
            
            mu[:,j] = (np.sum(np.multiply(np.broadcast_to(Pij[:,j],data.shape),data),axis = 1))/(np.sum(Pij[:,j])+realmin)
            
            a = data - mu[:,j].reshape(mu[:,j].shape[0],1)
            
            print(a.shape)#1200,1000
#            
#            sig[:,:,j] = Pij[:,j]
            #Can be a huge load to calculate sigma, so sticking with the sigma obtained from k-mean method
            #sig[:,:,j] = np.sum([Pij[i,j]*np.outer(a[:,i],a[:,i]) for i in range(a.shape[1])],axis=1)/(np.sum(Pij[:,j])+realmin)
            #print(sig.shape)
#            for i in range(a.shape[1]):
#                sigsum = sigsum + Pij[i,j]*np.matmul(a[:,i].reshape(a.shape[0],1),np.transpose(a[:,i].reshape(a.shape[0],1)))
#            sig[:,:,j] = sigsum/(np.sum(Pij[:,j])+realmin)   
#        
        loglik_new = 0
        for j in range(prior.shape[1]):
            loglik_new = loglik_new + prior[0,j]*gaussPDF.logGaussPdf(data,mu[:,j].reshape(mu.shape[0],1),sig[:,:,j])
        print('Main Loop Time: {}'.format(time.time()-start_main_loop))
        
        loglik = np.sum(loglik_new)
        
        prior = np.nan_to_num(prior)
        mu = np.nan_to_num(mu)
        sig = np.nan_to_num(sig)
        
        if np.absolute(loglik-loglik_old) <= loglik_threshold:#checking for the contingency
            return prior,mu,sig
        
        loglik_old = loglik
        
        
        
    return prior,mu,sig
    
    
            
    

            
        
    
        
    