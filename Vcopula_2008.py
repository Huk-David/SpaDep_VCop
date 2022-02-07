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
from itertools import combinations

# Based on [1]Bárdossy, A., and Li, J. (2008), Geostatistical interpolation using copulas, Water Resour. Res., 44, W07412, doi:10.1029/2007WR006115. 

class GP_Kernel:
    #working!
    #print(GP_Kernel().Exp_Quad())
    '''
    Contains different possible choices for GP kernels. These are to be used for the correlation matrix of [1].
    '''
    def __init__(self,x1=A([0,0]),x2=A([1,0]),theta=[1,1,1,1]): 
        '''
        x1,x2: [longitude,latitude] of locations 1 and 2 to get the correlation for.
        ''' 
        self.x1=x1
        self.x2=x2
        self.theta=theta               

    def W_Noise(self):
        '''
        x1,x2: [longitude,latitude] of locations 1 and 2 to get the correlation for.
        '''
        if (self.x1[0]==self.x2[0]) and (self.x1[1]==self.x2[1]): 
            return 1
        else:
            return 0
        
    def Exp_Quad(self):
        '''
        l: lenghtscale
        x1,x2: [longitude,latitude] of locations 1 and 2 to get the correlation for.
        '''
        l=self.theta[1]
        h = np.linalg.norm(self.x1-self.x2)
        return np.exp( (-h)/(2*(l**2))  )  

    def Rat_Quad(self):
        '''
        l: lenghtscale
        a: scale mixture (a>0)
        x1,x2: [longitude,latitude] of locations 1 and 2 to get the correlation for.
        '''
        l=self.theta[1]
        a=self.theta[2]
        if a<0:
            warnings.warn('Rat_Quad parameter a is not >0.')
        h = np.linalg.norm(self.x1-self.x2)
        return ( 1+(  ( (h)/(2*a*(l**2))  )  ) )**(-a)  
 
class Vcop08:
    '''
    Used to simulate data from or estimate parameters for a Vcopula.
    '''
    def __init__(self,n,locations,gpk):
        '''
        n: number of locations. Each location corresponds to one observation output per time unit.
        locations: array of [longitude,lattitude] for each of the n sites.
        gpk: Gaussian process kernel to be used as correlartion function. One from ('W_Noise','Exp_Quad','Rat_Quad').
        '''
        self.n=n
        self.locations=locations
        self.gpk=gpk
        
    def cor_f(self,loc1,loc2,theta): 
        #working
        '''
        Correlation function.
        loc1,loc2: locations for first and second variable.
        '''   
        if self.gpk=='W_Noise':
            return GP_Kernel(loc1,loc2,theta).W_Noise()
        elif self.gpk=='Exp_Quad':
            return GP_Kernel(loc1,loc2,theta).Exp_Quad()
        else:
            return GP_Kernel(loc1,loc2,theta).Rat_Quad()


    def simulate(self,theta,sets):
        #working
        #Vcop08(3,A([[0,0],[1,0],[0,1]]),'Exp_Quad').simulate([1,1,1,1],4)
        '''
        Function to simulate m sets of uniform [0,1] realisatoins for each location.
        theta: parameter vector [l,alpha,k,m] where l,alpha are used in the correlation matrix, and k,m in the vtransform.
        sets: natural number corresponding to the number of simulated sets of n realisations.
        returns: Z=m sets of n Vtransformed variables, and U=m sets of n uniform[0,1] vtransformed variables
        '''
        k=theta[2]
        m=theta[3]
        #1 - Construct Sigma - the correlation matrix
        n=self.n
        loc=self.locations
        sigma=np.identity(n)
        for couple in combinations(range(n),2):
            i=couple[0]
            j=couple[1]
            cor=self.cor_f(loc[i],loc[j],theta)
            sigma[i][j]=cor
            sigma[j][i]=cor
        
        #2,3 - Generate 'sets' sets of N_n(0,sigma) variables, and Apply Vtransform to Y variables
        #4 - Go from Z to u - uniform [0,1] variables
        U=[] # m sets of rows of n unif vtransforms
        Z=[]
        for i in range(sets):
            y_temp=mvn.rvs(mean=np.zeros(n)  ,cov= sigma)
            U.append(np.zeros(n))
            Z.append(np.zeros(n))
            for (j,y) in enumerate(y_temp):
                #Vtransform
                if y<m:    
                    U[i][j]=m-y
                else:
                    U[i][j]=k*(y-m) 
                Z[i][j]=U[i][j]
                #Take Z[i][j] to U[0,1], by applying H1
                U[i][j]=scs.norm.cdf((U[i][j]/k)+m)-scs.norm.cdf(m-U[i][j])
            #Z=[[set 1],...,[set 'sets']] and y[i]=[set i]=[obs 1,...,obs n] and y[i][j]=jth obs in ith set.
        out=[Z,U]
        return out
        
    def estimate(self,V_data='none',U_data='none'):
        '''
        A functoin to evaluate the likelihood for a given set of data.
        V_data: Vtransformed variables. If not given, evaluation will be done using U_data instead.
        U_data: uniform[0,1] Vtransformed variables.If not given, evaluation will be done using V_data instead.
        '''
        if U_data=='none':
            print('V')
        else:
            print('U')
        
        
Vcop08(2,1,1).estimate(U_data=[0])

