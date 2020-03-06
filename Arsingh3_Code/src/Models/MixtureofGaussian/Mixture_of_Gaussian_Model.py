# -*- coding: utf-8 -*-
import scipy.special as sc
import numpy as np
import src.Models.MixtureofGaussian.EM_MOG as EM
import src.EM_init as kmean_init
import src.Distributions.gaussPDF as g
import src.activation as act
import pandas as pd
import json
import sys
realmin = sys.float_info[3]

class GMM(object):
    
    def __init__(self,nbmixtures):
        '''Parameters:
        nbmixtures: Number of mixtures'''
        
        self.nbmixtures = nbmixtures
        self.mu = None
        self.sigma = None
        self.prior = None
        
    def fit(self,data,iterations):
        #Calculate the initial guess using Kmean method
        prior,mu,sigma = kmean_init.EM_init(data,self.nbmixtures)
        #Calculate the actual prior, mu and sigma
        self.prior,self.mu,self.sigma = EM.EM(data,prior,mu,sigma,iterations=iterations)
    
    def save(self,filename):
        datasave = {'Clusters':self.nbmixtures,
                    'Mu':self.mu,
                    'Sigma':self.sigma,
                    'Prior':self.prior}
        f = open(filename,'w')
        json.dump(datasave,f)
        f.close()
        
        
    
    
    
    
def MOGBinClassifier(x,mu1,mu2,sigma1,sigma2,prior1,prior2,t = 0.5):
    '''
    Parameters:
    x: the input data is a numpy array of shape (featurelength,datapoints) model is evaluated on (1200,1000)
    mu1,mu2: The mean of the classes which needs to be classified, shape->(1200,k=number of clusters)
    sigma1,sigma2: the variance of the calsses which needs to be classified
    prior1,prior2 are the priors of the MOG for classes to be classified 
    
    Return:
    returns a lsit and a numoy array
    '''
    p1 = np.ndarray((x.shape[1],prior1.shape[1]))
    #np.zeros((prior1.shape)) 
    p2 = np.ndarray((x.shape[1],prior1.shape[1]))
    print(prior1.shape)
    print(prior1.shape[0])
    
    
    
    for i in range(prior1.shape[1]):
        p1[:,i] = np.nan_to_num((np.log(prior1[0,i]+realmin) + g.logGaussPdf(x,mu1[:,i].reshape(mu1.shape[0],1),sigma1[:,:,i]))).reshape((p1.shape[0]))
    for i in range(prior2.shape[1]):
        p2[:,i] = np.nan_to_num((np.log(prior2[0,i]+realmin) + g.logGaussPdf(x,mu2[:,i].reshape(mu2.shape[0],1),sigma2[:,:,i]))).reshape((p1.shape[0]))
    
    #p1 -> number of data,number of gaussian
    #p2 -> number of data, number of gaussian
    p1 = np.nan_to_num(sc.logsumexp(p1,axis=1))
    p2 = np.nan_to_num(sc.logsumexp(p2,axis=1))
    
    pred = []
    
    df1 = pd.DataFrame(p1)
    df2 = pd.DataFrame(p2)
    df1.to_csv('p1data.csv')
    df2.to_csv('p2data.csv')
    
    P1 = p1/(p1+p2)
    P2 = p2/(p1+p2)
    df1 = pd.DataFrame(P1)
    df2 = pd.DataFrame(P2)
    df1.to_csv('p1data.csv')
    df2.to_csv('p2data.csv')
    #normalizer = 1e12 #To normalize the log-likelihood
    
    P1 = P1.reshape((p1.shape[0]))
    P2 = P2.reshape((p2.shape[0]))
    
    
    for i in range(P1.shape[0]):
        if (P2[i]-P1[i])>0:
            pred.append(1)
        else:
            pred.append(0)
    
    
    sf = 100       
    
    return pred, act.sigmoid(sf*(P1-P2))
    
 
##### Loading a Model
def load(filename):
    
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    MOG = GMM(data['Clusters'])
    MOG.mu = data['Mu']
    MOG.sigma = data['Sigma']
    MOG.prior = data['Prior']
    return MOG       