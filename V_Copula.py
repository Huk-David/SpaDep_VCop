#Code used to simulate and estimate parameters for a V-copula. Based on [1]Bárdossy, A.: Interpolation of groundwater quality parameters with some values below the detection limit, Hydrol. Earth Syst. Sci., 15, 2763–2775, https://doi.org/10.5194/hess-15-2763-2011, 2011
import numpy as np
from numpy import transpose as T
from numpy import array as A
from numpy.linalg import det as D
from numpy.linalg import inv as I
import matplotlib.pyplot as plt
import scipy.stats as scs
from math import *
from itertools import product
from scipy.stats import multivariate_normal as mvn
import warnings

class Vcop_sim():
    '''
    Used to simulate data. The simulated data will then be used to test the MLE estimation for parameters used for the simulation.
    '''

    def _init_(self,n,gp_kernel='W_Noise',theta,grid):                 
        '''
         n: number of obs
         gp_kernel: kernel to be used
         theta: parameters for gp_kernel, as a vector
         grid: a vector of [longitude.latitude] for each location.
        '''
        self.n=n
        self.theta=theta
        self.grid=grid

    class GP_Kernel():
        '''
        Contains different possible choices for GP kernels. These are to be used for the correlation matrix of [1].
        '''
        def W_Noise(n,theta,x1,x2):
            '''
            sig: The variance of the noise is sigma**2.
            x1,x2: [longitude,latitude] of locations 1 and 2 to get the correlation for.
            '''
            sig=theta[0]
            if x1==x2: 
                return sig**2
            else:
                return 0
        
        def Exp_Quad(n,theta,l,x1,x2):
            '''
            sig: The variance for a single obs is sigma**2.
            l: lenghtscale
            x1,x2: [longitude,latitude] of locations 1 and 2 to get the correlation for.
            '''
            sig=theta[0]
            l=theta[1]
            h = numpy.linalg.norm(x1-x2)
            return (sig**2)*np.exp( (-h)/(2*(l**2))  )  

        def Rat_Quad(n,theta,x1,x2):
            '''
            sig: The variance for a single obs is sigma**2.
            l: lenghtscale
            a: scale mixture (a>0)
            x1,x2: [longitude,latitude] of locations 1 and 2 to get the correlation for.
            '''
            sig=theta[0]
            l=theta[1]
            a=theta[2]
            if a<0:
                warnings.warn('Rat_Quad parameter a is not >0.')
            h = numpy.linalg.norm(x1-x2)
            return (sig**2)*(  ( 1+(  ( (h)/(2*a*(l**2))  )  ) )**(-a)  )


