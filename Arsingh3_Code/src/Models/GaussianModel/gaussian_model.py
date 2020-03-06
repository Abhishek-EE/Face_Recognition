# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 23:40:43 2020

@author: abhis
"""
import numpy as np
import pandas as pd
import src.activation as act

class Gauss_Model_bin_Classifier(object):
    
    '''Some work needs to be done here
    Gaussian model for binary classification'''
    def __init__(self,shape):
        '''Something needs to be written here'''
        self.shape = shape
        self.mean_class1 = np.zeros(shape)
        self.var_class1 = np.matmul(self.mean_class1,np.transpose(self.mean_class1))
        self.mean_class2 = np.zeros(shape)
        self.var_class2 = np.matmul(self.mean_class1,np.transpose(self.mean_class1))
        
        
    def parameter_estimator(self,training_data_class1,training_data_class2):
        '''takes in a pandas dataframe and uses it to estimate mean and variance'''
#        self.mean_class1 = np.mean(training_data_class1.Feature,axis =0)
#        self.mean_class2 = np.mean(training_data_class1.Feature,axis = 0)
#        
        self.mean_class1 = self.mean_estimator(training_data_class1)
        self.var_class1 = self.var_estimator(training_data_class1,self.mean_class1)
        self.mean_class2 = self.mean_estimator(training_data_class2)
        self.var_class2 = self.var_estimator(training_data_class2,self.mean_class2)
        
        
        '''Really need to find out how it will work'''
        '''Training data is a list of feature vectors'''

    def mean_estimator(self,x):
        '''Takes in the a pandas dataframe of feature vectors and returns the mean vector'''
        add = np.zeros((x.Feature[0].shape))
        for i in range(len(x.Feature)):
            add = add + x.Feature[i]
        return add/len(x.Feature)
    
    def var_estimator(self,x,mu):
        '''Takes in a pandas dataframe, estimated mean vector and returns a diagonal variance matrix of type array'''
        sq_sum = np.zeros(x.Feature[0].shape)
        for i in range(len(x.Feature)):
            sq_sum = (x.Feature[i]-mu)**2
        diagmat = np.zeros(sq_sum.shape)
        diagmat = np.matmul(diagmat,np.transpose(diagmat))
        for i in range(sq_sum.shape[0]):
            diagmat[i][i] = sq_sum[i]
        return diagmat/(len(x.Feature)-1)
    
    
    def classifier(self,feature_vector, a, t):
        '''Is this all I need? I think  I know how to do this one at least'''
        
        #a = np.log(np.linalg.det(self.var_class1))-np.log(np.linalg.det(self.var_class2))#-> Ideally want to do this but due to overflow using log difference
        
        norm_const = 1e9 #Introducing a normalization constant to take care of overflow/underflow
        b = np.matmul(np.transpose(feature_vector-self.mean_class1),np.linalg.inv(self.var_class1)@(feature_vector-self.mean_class1))
        c = np.matmul(np.transpose(feature_vector-self.mean_class2),np.linalg.inv(self.var_class1)@(feature_vector-self.mean_class2))
#        b = np.matmul(np.transpose(feature_vector-self.mean_class1),(feature_vector-self.mean_class1))
#        c = np.matmul(np.transpose(feature_vector-self.mean_class2),(feature_vector-self.mean_class2))
#        
        print(act.sigmoid((a+b-c)/norm_const))
        #print(a)
        #print(type(a))
        #print(c)
        if act.sigmoid((a+b-c)/norm_const) <= t:
            return 1,act.sigmoid((a+b-c)/norm_const)
        else:
            return 0,act.sigmoid((a+b-c)/norm_const)
        
        
    def log_det_diag(self,inputmat):
        '''Calculates the logarithmic deteminant of a diagonal Matrix
        Takes in a 2-D symmetric array and returns a number
        Uses '''
        lndet = 0
        #To ignore 
        #print(np.log(inputmat[5][5]))
        for i in range(inputmat.shape[0]):
            lndet = lndet + np.log(inputmat[i][i]+1) 
        #print(lndet)
        return lndet
            
    def predictions(self,data, t = 0.5):
        'Takes in a pandas dataframe and returns a list of classified output' 
        pred = []
        prob = []
        a = self.log_det_diag(self.var_class1) - self.log_det_diag(self.var_class2)
        for i in range(len(data.Feature)):
            predvalue, probvalue = self.classifier(data.Feature[i],a,t)
            pred.append(predvalue)
            prob.append(probvalue)
        return pred,prob

    def save(self, filename):
        '''To Save the Parameters'''
        x = {
             'Mean1':self.mean_class1,
             'Mean2':self.mean_class2,
             'Variance1':self.var_class1,
             'Variance2':self.var_class2}
        df = pd.DataFrame(x, index = False)
        df.to_csv(filename)
        
    
    
    
def load(filename):
    '''A function to load a gaussian model from the input'''
    df = pd.read_csv(filename)
    shape = (1200,1)
    model = Gauss_Model_bin_Classifier(shape)
    model.mean_class1 = df.Mean1
    model.mean_class2 = df.Mean2
    model.var_class1 = df.Variance1
    model.var_class2 = df.Variance2
    
    
