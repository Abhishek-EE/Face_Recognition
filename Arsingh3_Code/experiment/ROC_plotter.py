# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 04:54:04 2020

@author: Abhishek Ranjan Singh
"""
import matplotlib.pyplot as plt
import numpy as np

class ROC(object):
    
    def __init__(self,shape):
        '''I don't think I need this but anyways'''
    
    def ConfusionMat_calculator(self,predictionResult):
        '''Takes in a dataframe and gives out a list'''
        
        Addop = predictionResult.IsFace + predictionResult.Predictions
        Subop = predictionResult.IsFace - predictionResult.Predictions
        Tp = 0
        Fp = 0
        Fn = 0
        Tn = 0
        for i in range(len(Addop)):
            if Addop[i] == 0:
                Tn = Tn+1
            if Addop[i] == 2:
                Tp = Tp+1
            if Subop[i] == -1:
                Fp = Fp+1
            if Subop[i] == 1:
                Fn = Fn+1
        return Tp,Tn,Fp,Fn
    
    def TPR_calculator(self,predictionResult):
        Tp,Tn,Fp,Fn = self.ConfusionMat_calculator(predictionResult)
        return (Tp/(Tp+Fn))
    
    def FPR_calculator(self,predictionResult):
        Tp,Tn,Fp,Fn = self.ConfusionMat_calculator(predictionResult)
        return (Fp/(Tn+Fp))
                
    def prob_to_Pred(self,predictionResult,t):
        for i in range(len(predictionResult.Probability)):
            if predictionResult.Probability[i] <= t: # Ideally we check for > sign but in this case < because when we multiply by - in the case of bayesian classifier the sign changes and I have used the positive diff
                predictionResult.at[i,'Predictions'] = 1
            else:
                predictionResult.at[i,'Predictions'] = 0
        return predictionResult
    
    def Classifier_Details(self,predictionResult):
        Tp,Tn,Fp,Fn = self.ConfusionMat_calculator(predictionResult)
        m_r = (Fp + Fn)/(Tp+Tn+Fp+Fn)
        FNR = (Fn/(Tp+Fn))
        FPR = (Fp/(Fp+Fn))
        print('Misclassification Rate = {}'.format(m_r))
        print('False Negative Rate = {}'.format(FNR))
        print('False Positive Rate = {}'.format(FPR))
        
    def ROC_plotter(self,predictionResult,label):
        TPR = []
        FPR = []
        traverse = np.linspace(0,1,100)
        for t in traverse:
            predictionResult = self.prob_to_Pred(predictionResult,t)
            TPR.append(self.TPR_calculator(predictionResult))
            FPR.append(self.FPR_calculator(predictionResult))
        ROcurve, = plt.plot(FPR,TPR,'r-')
        RandomGuess, = plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate or (1 - Specifity)')
        plt.ylabel('True Positive Rate or (Sensitivity)')
        plt.legend([ROcurve,RandomGuess],[label,'RandomGuess'],loc="lower right")
        plt.show()

