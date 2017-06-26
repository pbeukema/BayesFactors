import numpy as np
import math
import scipy.integrate as integrate
from __future__ import division


def jzs_corbf(r,n):
    # This function computes the Bayes Factor to evaluate evidence for or against a linear relationship
    # H0: no linear relationship
    # H1: linear relationship exists
    # For interpretation of the result see below

    # Usage 
    # r = correlation_coefficient from sample
    # n = n_samples (e.g. datapoints)
    # BF10 = jzs_corbf(r,n)
    
    # Interpretation:
    #       >   100   Decisive evidence for H1
    # 30    -   100   Very Strong evidence for H1
    # 10    -   30    Strong evidence for H1
    # 3     -   10    Substantial evidence for H1
    # 1     -   3     Anecdotal evidence for H1
    # 1               No evidence
    # 1/3   -   1     Anecdotal evidence for H0
    # 1/10  -  1/3    Substantial evidence for H0
    # 1/30  -  1/10   Strong evidence for H0
    # 1/100 -  1/30   Very Strong evidence for H0
    #       <  1/100  Decisive evidence for H0

    # As described in Wetzels & Wagemakers 2012
    # Psychon Bull Rev. 2012 Dec; 19(6): 1057–1064.
    # doi:  10.3758/s13423-012-0295-x

    g1 = lambda g: (1 + g)**((n - 2)/2) * (1+(1-r**2)*g)**(-(n-1)/2) * g**(-3/2) * np.exp(-n/(2*g))
    BF10 = (np.sqrt(n/2)/math.gamma(1/2))* integrate.quad(g1, 0,np.inf)[0]
    return BF10



def eval_null(SSE1, SSE0, k1,k0, n)
    # Model Comparisons using SSE/SST:
    # To identify posterior probability of null 
    # It is often useful to consider the delta BIC10 quantity using BF01
    # Note that BF01 != BF10
    
    # Usage
    # SSE1: sum of squared errors for H1 (alternative)
    # SSE0: sum of squared errors for H0 (null)
    # k1: # parameters for alternative model
    # k0: # parameters for null model
    # n:number of subjects
    
    # BF01  
    #       >   100   Decisive evidence for H1
    # 30    -   100   Very Strong evidence for H1
    # 10    -   30    Strong evidence for H1
    # 3     -   10    Substantial evidence for H1
    # 1-3      Anecdotal evidence for H1


    # As described in Wetzels & Wagemakers 2012
    # Psychon Bull Rev. 2012 Dec; 19(6): 1057–1064.
    # doi:  10.3758/s13423-012-0295-x
    
    deltaBIC10 = n*np.log(SSE1/SSE0) + (k1 - k0)*np.log(n)
    BF01 = np.exp(deltaBIC10/2)
    return BF01


    
