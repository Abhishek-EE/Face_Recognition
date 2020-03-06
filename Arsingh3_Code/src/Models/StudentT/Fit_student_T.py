# -*- coding: utf-8 -*-
import numpy as np
from scipy.special import psi, gammaln
#import fit_t_cost as ftc
import src.Models.StudentT.Fit_T_cost as ftc

def fit_t(x, precision):
    print('Performing fit_t...')
#    x = np.random.multivariate_normal(da);
    (I,D) = np.shape(x)
    # Initialize the Mean
    #dataset_mean = np.divide(np.sum(x,axis = 0),I)
    dataset_mean = x.mean(axis=0)
    mu = np.reshape(dataset_mean,(1,-1))
   
    #Initialize sig to the covariance of the dataset.
    dataset_variance = np.zeros([D, D])
    x_minus_dataset_mean = np.subtract(x, dataset_mean)
    for i in range(I):
        mat = np.reshape(x_minus_dataset_mean[i,:],(1,-1))
        mat = np.dot(np.transpose(mat),mat)
        dataset_variance = dataset_variance + mat;
    sig = np.divide(dataset_variance,I);

    ##Initialize degrees of freedom to 1000 (just a random large value).
    nu = 1
    ##The main loop.
    iterations = 0    
    previous_L = 1000000 # just a random initialization
    delta = np.zeros([I,1])
    #delta1 = np.zeros([I,1])
    while True:
        #Expectation step.
        #Compute delta.
        x_minus_mu = np.subtract(x, mu)
        temp = np.dot(x_minus_mu,np.linalg.inv(sig))
        for i in range(I):
            delta[i] = np.dot(np.reshape(temp[i,:],(1,-1)),np.transpose(np.reshape(x_minus_mu[i,:],(1,-1))))
           
        # Compute E_hi.
        nu_plus_delta = nu + delta
        E_hi = np.divide((nu + D),nu_plus_delta)
        ## Compute E_log_hi.
        E_log_hi = psi((nu+D)/2) - np.log(nu_plus_delta/2);
     
        ## Maximization step.
        ## Update mu.
               
        E_hi_sum = np.sum(E_hi)
        E_hi_times_xi = E_hi * x
        mu = np.reshape(np.sum(E_hi_times_xi, axis=0),(1,-1))
        mu = np.divide(mu,E_hi_sum)
        ## Update sig.
        x_minus_mu = np.subtract(x, mu)
        sig = np.zeros([D,D])
        for i in range(I):
            xmm = np.reshape(x_minus_mu[i,:],(1,-1))
            sig = sig + (E_hi[i] * np.dot(np.transpose(xmm),xmm))
        sig = sig / E_hi_sum
       
        #Update nu by minimizing a cost function with line search.
        nu = ftc.fit_t_cost(E_hi,E_log_hi)
       
        ## Compute delta again, because the parameters were updated.
        temp = np.dot(x_minus_mu,np.linalg.inv(sig))
        # temp1 = np.linalg.inv(sig)
        for i in range(I):
            delta[i] = np.dot(np.reshape(temp[i,:],(1,-1)),np.transpose(np.reshape(x_minus_mu[i,:],(1,-1))))
           
        ## Compute the log likelihood L.
        (sign, logdet) =  np.linalg.slogdet(np.array(sig))
        L = I * (gammaln((nu+D)/2) - (D/2)*np.log(nu*np.pi) - logdet/2 - gammaln(nu/2))
        s = np.sum(np.log(1 + np.divide(delta,nu))) / 2
        L = L - (nu+D)*s;
        iterations = iterations + 1;
        print(str(iterations)+' : '+str(L))
        if (np.absolute(L - previous_L) < precision) or iterations == 100:
            break
        previous_L = L;
    return(mu,sig,nu)


