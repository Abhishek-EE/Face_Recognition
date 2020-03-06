# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 04:54:04 2020

@author: Abhishek Ranjan Singh
"""
import sys
import pandas as pd
import argparse
import cv2
import ROC_plotter as ROC
import numpy as np
from sklearn.decomposition import PCA
sys.path.append('../')
Data_Path = ('..//..//Data//')
import mu_sig_plot
import src.FeatureExtraction as feature
import src.Models.GaussianModel.gaussian_model as gaussian_bin
import src.Models.MixtureofGaussian.Mixture_of_Gaussian_Model as MOG
import src.Models.FactorAnalyser.FactorAnalyser as FA
import src.Distributions.tdist as T
import src.Models.StudentT.Fit_student_T as Stud
import src.activation as act



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', action='store_true', help='Check data loading.')
    parser.add_argument('--trainMOG', action='store_true', help='Train Mixture of Gaussian')
    parser.add_argument('--trainFA', action='store_true',help='Factor Analysis')
    parser.add_argument('--traingaussian', action='store_true',help='Train the Gaussian model')
    parser.add_argument('--trainT', action='store_true',help='Train T dist')
    return parser.parse_args()

def load_data():
    dfFace = pd.read_csv(Data_Path + 'faceData.csv')
    dfTrainFace = image_to_feature(dfFace[0:1000].copy())
    dfTestFace = image_to_feature(dfFace[1520:1620].copy())
    
    dfNonFace = pd.read_csv(Data_Path + 'NonFaceData.csv')
    dfTrainNF = image_to_feature(dfNonFace[0:1000].copy())
    dfTestNF = image_to_feature(dfNonFace[1520:1620].copy())
    
    dfTrainFace.to_csv('predictions/trainF_feature.csv',index = False)
    dfTrainNF.to_csv('predictions/trainNF_features.csv', index = False)
    
  
    print('Number of training data for face: {}'.format(len(dfTrainFace)))
    print('Number of testing data for face: {}'.format(len(dfTestFace))) 
    print('Number of training data for Nonface: {}'.format(len(dfTrainNF)))
    print('Number of testing data for Nonface: {}'.format(len(dfTestNF)))

    return dfTrainFace, dfTestFace, dfTrainNF, dfTestNF

def image_to_feature(df):
    'takes a df and converts it to images'
    f = []
    for i in df.Image:
        image = cv2.imread((Data_Path + i).replace('\\','//'))
        f.append(feature.feature_extraction(img = image))
    df['Feature'] = f     
    return df

def dataprep(trainf):
    data = np.zeros((trainf.Feature.to_numpy()[0].shape[0],(trainf.Feature.to_numpy().shape[0])))
    for i in range(data.shape[1]):
        data[:,i] = np.reshape(trainf.Feature.to_numpy()[i],(trainf.Feature.to_numpy()[i].shape[0]))
    return data


def gaussian():
    trainf,testf,trainNF,testNF = load_data()
    model = gaussian_bin.Gauss_Model_bin_Classifier((1200,1))
    model.parameter_estimator(trainf,trainNF)
    df = pd.concat([testf,testNF],ignore_index = True)
    store_pred,store_prob = model.predictions(df)
    print(len(store_pred))
    print(df.shape)
    df['Predictions'] = store_pred
    df['Probability'] = store_prob
    df.to_csv('predictions/prediction_gaussian.csv')
    roc = ROC.ROC(1200)
    roc.Classifier_Details(df)
    roc.ROC_plotter(df,'GaussianClassifier')
    mu_sig_plot.plot_mu_sig(model.mean_class1,model.var_class1,'gaussianfacemean','gaussianfacevar','RGB',20,'BOTH')
    mu_sig_plot.plot_mu_sig(model.mean_class2,model.var_class2,'gaussiannonfacemean','gaussiannonfacevar','RGB',20,'BOTH')
    
    

def MixtureOfGaussian():
    trainf,testf,trainNF,testNF = load_data()
    trainfp,testfp,trainNFp,testNFp = dataprep(trainf),dataprep(testf),dataprep(trainNF),dataprep(testNF)

    #MOG
    mogface = MOG.GMM(10)
    mogface.fit(trainfp,10)
    print('Training Done for Face')
    print(mogface.mu.shape)
    print(mogface.sigma.shape)
    atmudf = pd.DataFrame(mogface.mu)
    atmudf.to_csv('aftermean.csv')
    afprior = pd.DataFrame(mogface.prior)
    afprior.to_csv('finalprrior.csv')
    
    mognonface = MOG.GMM(10)
    mognonface.fit(trainNFp,10)
    print('Training Done for NonFace')
    df = pd.concat([testf,testNF],ignore_index = True)
    tnp = np.concatenate((testfp,testNFp),axis = 1)
    store_pred,store_prob = MOG.MOGBinClassifier(tnp,mogface.mu,mognonface.mu,mogface.sigma,mognonface.sigma,mogface.prior,mognonface.prior)
    print(store_prob.shape)
    print(len(store_pred))
    print(df.shape)
    df['Predictions'] = store_pred
    df['Probability'] = store_prob
    df.to_csv('predictions/prediction_MOG.csv')
    roc = ROC.ROC(1200)
    roc.Classifier_Details(df)
    roc.ROC_plotter(df,'MOG Classifier')
    for i in range(3):     
        mu_sig_plot.plot_mu_sig(mogface.mu[:,i],mogface.sigma[:,:,i],'MOGfacemean'+str(i),'MOGfacevar'+str(i),'RGB',20,'BOTH')
        mu_sig_plot.plot_mu_sig(mognonface.mu[:,i],mognonface.sigma[:,:,i],'MOGnonfacemean'+str(i),'MOGnonfacevar'+str(i),'RGB',20,'BOTH')


def factoranalyser():
    trainf,testf,trainNF,testNF = load_data()
    trainfp,testfp,trainNFp,testNFp = dataprep(trainf),dataprep(testf),dataprep(trainNF),dataprep(testNF)
    #Factor Analysis
    FAface = FA.FactorAnalysis(3)
    FAface.fit(trainfp,5)
    print('Dimensions of phi: {}'.format(FAface.phi.shape))
    print('Dimensions of mu: {}'.format(FAface.mu.shape))
    print('Dimensions of sigma: {}'.format(FAface.sigma.shape))
    print('Face Training Done')
    FAnonface = FA.FactorAnalysis(3)
    FAnonface.fit(trainNFp,5)
    print('Dimensions of phi: {}'.format(FAface.phi.shape))
    print('Dimensions of mu: {}'.format(FAface.mu.shape))
    print('Dimensions of sigma: {}'.format(FAface.sigma.shape))
    print('Non-Face Training Done')
    df = pd.concat([testf,testNF],ignore_index = True)
    tnp = np.concatenate((testfp,testNFp),axis = 1)
    store_pred,store_prob = FA.FAbinClassifier(tnp,FAface.mu,FAnonface.mu,FAface.sigma,FAnonface.sigma,FAface.phi,FAnonface.phi)
    
    print(store_prob.shape)
    print(len(store_pred))
    print(df.shape)
    df['Predictions'] = store_pred
    df['Probability'] = store_prob
    df.to_csv('predictions/prediction_FA.csv')
    
    #Plotting ROC
    roc = ROC.ROC(1200)
    roc.Classifier_Details(df)
    roc.ROC_plotter(df,'FactorAnalyser')
    
    #Visualaizing Mu and Sigma
    
    sigmaclass1 = np.matmul(FAface.phi,np.transpose(FAface.phi)) + np.diag(FAface.sigma)
    sigmaclass2 = np.matmul(FAnonface.phi,np.transpose(FAnonface.phi)) + np.diag(FAnonface.sigma)
    
    mu_sig_plot.plot_mu_sig(FAface.mu,sigmaclass1,'FAfacemean','FAfacevar','RGB',20,'BOTH')
    mu_sig_plot.plot_mu_sig(FAnonface.mu,sigmaclass2,'FAnonfacemean','FAnonfacevar','RGB',20,'BOTH')

    

def data_pca(temp1):
    data=temp1.Feature.to_numpy()
    array_data = np.zeros((len(data),data[0].shape[0]))
    for i in range(len(data)):
        array_data[i,:] = data[i].flatten()
       
    pca_face=PCA(30)    
    data_pca_format=pca_face.fit_transform(array_data)
    return data_pca_format,array_data


def StudnetT():
   
    trainface,testface,trainnonface,testnonface=load_data()
    pca_face=PCA(30)
    f_train_pca,array_trainf = data_pca(trainface)
    f_test_pca,array_testf = data_pca(testface)
    nf_train_pca,array_trainnf = data_pca(trainnonface)
    nf_test_pca,array_testnf = data_pca(testnonface)

    (mu_f,sig_f,nu_f)= Stud.fit_t(f_train_pca,0.01)
    (mu_nf,sig_nf,nu_nf)= Stud.fit_t(nf_train_pca,0.01)
   
                               
    px_face_pf = T.StudTpdf(f_test_pca,mu_f,sig_f,nu_f)
    px_nonface_pf = T.StudTpdf(nf_test_pca,mu_f,sig_f,nu_f)
    px_face_pnf = T.StudTpdf(f_test_pca,mu_nf,sig_nf,nu_nf)
    px_nonface_pnf = T.StudTpdf(nf_test_pca,mu_nf,sig_nf,nu_nf)

    Prob_face = np.concatenate((px_face_pf,px_nonface_pf))
    Prob_nonface = np.concatenate((px_face_pnf,px_nonface_pnf))

    
   
    df = pd.concat([testface,testnonface],ignore_index = True)
    print(Prob_face.shape)
    pred = []
    for i in range(Prob_face.shape[0]):
        if Prob_face[i] > Prob_nonface[i]:
            pred.append(1)
        else:
            pred.append(0)
    Prob_faceact = Prob_face/(Prob_face+Prob_nonface)
    Prob_nonfaceact = Prob_nonface/(Prob_face+Prob_nonface)

    df['Predictions'] = pred
    df['Probability'] = act.sigmoid(5*(Prob_nonfaceact-Prob_faceact))
   
    df.to_csv('studentT.csv')
   
    roc = ROC.ROC(1200)
    roc.Classifier_Details(df)
    roc.ROC_plotter(df,'StudentT')
    print(mu_f.shape)
    print(sig_nf.shape)
   
#    mu_f = pca_face.inverse_transform(mu_f.reshape((mu_f.shape[1],mu_f.shape[0])))
#    mu_nf = pca_face.inverse_transform(mu_nf.reshape((mu_nf.shape[1],mu_nf.shape[0])))
#    sig_f = pca_face.inverse_transform(sig_f)
#    sig_nf = pca_face.inverse_transform(sig_nf)
#   
#    mu_sig_plot.plot_mu_sig(mu_f,sig_f,'Tfacemean','Tfacevar','RGB',20,'BOTH')
#    mu_sig_plot.plot_mu_sig(mu_nf,sig_nf,'Tnonfacemean','Tnonfacevar','RGB',20,'BOTH')
##    
##def MoT():
#    

 
    
    


if __name__ == '__main__':
    
    FLAGS = get_args()
  
    if FLAGS.input:
        load_data()
    if FLAGS.trainMOG:
        MixtureOfGaussian()
    if FLAGS.traingaussian:
        gaussian()
    if FLAGS.trainFA:
        factoranalyser()
    if FLAGS.trainT:
        StudnetT()