import numpy as np
import sys

realmin = sys.float_info[3]
def gaussPDF(data, Mu, Sigma):
    return np.exp(logGaussPdf(data, Mu, Sigma))+realmin #To ignore the divide by zero error
    
def logGaussPdf(data,Mu,Sigma):
    '''Vector is 1200,1000 array,Mu is 1200,k array,Sigma is 1200,1200,k array'''
    
    nbVar = data.shape[0]
    try:
        nbData = data.shape[1]
    except:
        nbData = 1
    a = (-nbVar/2)*np.log(2*np.pi)

    (sign,logdet) = np.linalg.slogdet(Sigma)
    c = data-Mu
    x = np.zeros((nbData,1))
    Sigma_inv = np.linalg.inv(Sigma)
    
    for i in range(nbData):
        unk = Sigma_inv@((c[:,i]).reshape(c.shape[0],1))
        x[i,0] = np.dot(np.transpose(c[:,i].reshape((c.shape[0],1))),unk)
    logpdf = a - (0.5)*logdet -(0.5*x)
    return logpdf


    