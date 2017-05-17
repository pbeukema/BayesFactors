import numpy as np
import math
import scipy.integrate as integrate
from __future__ import division

# This function computes the Bayes Factor to evaluate evidence for or against the null hypothesis
# for linear models

# Usage 
# r = correlation_coeffient from sample
# n = n_samples (e.g. datapoints)
# BF10 = jzs_corbf(r,n)

def jzs_corbf(r,n):
    g2 = lambda g: (1 + g)**((n - 2)/2) * (1+(1-r**2)*g)**(-(n-1)/2) * g**(-3/2) * np.exp(-n/(2*g))
    BF10 = (np.sqrt(n/2)/math.gamma(1/2))* integrate.quad(g2, 0,np.inf)[0]
    return BF10
    
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
