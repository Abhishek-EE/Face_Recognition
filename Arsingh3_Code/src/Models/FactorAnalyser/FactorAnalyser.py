# -*- coding: utf-8 -*-
import src.Models.FactorAnalyser.EM_FA as EM
import src.Distributions.gaussPDF as g
import src.activation as act
import pandas as pd
import json
import numpy as np
import scipy.special as sc
#import src.FeatureExtraction as FE


class FactorAnalysis(object):
    
    def __init__(self,nbfactors):
        '''Parameters:
        nbfactors: Number of factors'''
        
        self.nbfactors = nbfactors
        self.mu = None
        self.sigma = None
        self.phi = None
        
    def defaultdatainitializer(self,data):
        '''
        Self.sigma has a shape of 1200, and self.mu has a shape 1200,1
        '''
        self.mu = np.mean(data,axis = 1)
        x_minus_mean = np.subtract(data,self.mu.reshape(self.mu.shape[0],1))
        print('shape of x_minus_mean {}'.format(x_minus_mean.shape))
        self.sigma = np.sum((x_minus_mean)**2,axis = 1)
        np.random.seed(0)
        self.phi = np.random.randn(data.shape[0],self.nbfactors)
        
        
    def fit(self,data,iterations):
        #Calculate the initial guess using Kmean method
        self.defaultdatainitializer(data)
        #Calculate the actual prior, mu and sigma
        self.phi,self.mu,self.sigma = EM.EM(data,self.phi,self.mu,self.sigma,iterations=iterations)
    
    def save(self,filename):
        datasave = {'Clusters':self.nbfactors,
                    'Mu':self.mu,
                    'Sigma':self.sigma,
                    'Prior':self.prior}
        f = open(filename,'w')
        json.dump(datasave,f)
        f.close()
        
        
def FAbinClassifier(x,mu1,mu2,sigma1,sigma2,phi1,phi2,t=0.5):
    '''
    Parameters:
        x: the input data is a numpy array of shape (featurelenth,number of datapoints) model is evaluated on (1200,1000)
        mu1,mu2: The mean of classes which needs to be classified, shape ->(featurelength,)
        sigma1,sigma2 : the variance of the calsses which needs to be classified
        its a diagonal mat with shape ->(1200,) only diagonal elements are passed
        phi1,phi2: factor matrix with shape (1200,3)
    '''
    pred = []
    mu_Class1 = mu1.reshape(mu1.shape[0],1)
    mu_Class2 = mu2.reshape(mu2.shape[0],1)
    sigmaclass1 = np.matmul(phi1,np.transpose(phi1)) + np.diag(sigma1)
    sigmaclass2 = np.matmul(phi2,np.transpose(phi2)) + np.diag(sigma2)
    
#    x_minus_mu1 = x - mu_Class1.reshape((mu_Class1.shape[0],1))
#    x_minus_mu2 = x - mu_Class2.reshape((mu_Class2.shape[0],1))
#    
#    (sign,logdet1) = np.linalg.slogdet(sigma)
#    (sign,logdet2) = np.linalg.slogdet(simga)
#    inv_Sigma1
    p1 = g.logGaussPdf(x,mu_Class1,sigmaclass1)
    p2 = g.logGaussPdf(x,mu_Class2,sigmaclass2)
    
#    p1 = sc.softmax(p1)
#    p2 = sc.softmax(p2)
#    
#    p1 = sc.multivariate_normal.pdf(x,mu_Class1.reshape((1200)),sigmaclass1)
#    p2 = sc.multivariate_normal.pdf(x,mu_Class2.reshape((1200)),sigmaclass2)
    
    
    P1 = p1/(p1+p2)
    P2 = p2/(p1+p2)
    
    dfp1 = pd.DataFrame(P1)
    dfp2 = pd.DataFrame(P2)
    dfp1.to_csv('P1_FA_.csv')
    dfp2.to_csv('P2_FA_.csv')
    
    P1 = P1.reshape(p1.shape[0])
    P2 = P2.reshape(p2.shape[0])
    
    for i in range(p1.shape[0]):
        if P2[i]-P1[i] < 0:
            pred.append(1)
        else:
            pred.append(0)
    return pred, act.sigmoid(1000*(P2-P1))



####### Loading a Model
#def load(filename):
#    """Load a neural network from the file ``filename``.  Returns an
#    instance of Network.
#
#    """
#    f = open(filename, "r")
#    data = json.load(f)
#    f.close()
#    MOG = GMM(data['Clusters'])
#    MOG.mu = data['Mu']
#    MOG.sigma = data['Sigma']
#    MOG.prior = data['Prior']
#    return MOG 
#    