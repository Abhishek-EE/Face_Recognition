# -*- coding: utf-8 -*-
from scipy.special import gammaln
import numpy as np

def StudTpdf(x, mu, sig, nu):
    (I,D)= np.shape(x)
    #    % gammaln is used instead of gamma to avoid overflow.
    #    % gamma((nu+D)/2)/gamma(nu/2) == exp(gammaln((nu+D)/2)-gammaln(nu/2)).
    c = np.exp(gammaln((nu+D)/2) - gammaln(nu/2));
    det = np.prod(np.diag(sig))
    c = c / ((nu*np.pi)**(D/2) * np.sqrt(det));
    delta = np.zeros([I,1])
    x_minus_mu = np.subtract(x, np.reshape(mu,(1,-1)))
    temp = np.dot(x_minus_mu,np.linalg.inv(np.diag(np.diag(sig))))
    for i in range(I):
        delta[i] = np.dot(np.reshape(temp[i,:],(1,-1)),np.transpose(np.reshape(x_minus_mu[i,:],(1,-1))))
    px = 1 + (delta / nu);
    px = px**((-nu-D) / 2);
    px = np.dot(px,c)
    return(px)
