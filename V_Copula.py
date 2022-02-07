#Code used to simulate and estimate parameters for a V-copula. Based on [1]Bárdossy, A.: Interpolation of groundwater quality parameters with some values below the detection limit, Hydrol. Earth Syst. Sci., 15, 2763–2775, https://doi.org/10.5194/hess-15-2763-2011, 2011

from ast import arg
import numpy as np
from numpy import transpose as T
from numpy import array as A
from numpy.linalg import det as D
from numpy.linalg import inv as I
import scipy.stats as scs
from math import *
from itertools import combinations
from scipy.stats import multivariate_normal as mvn
import warnings
from scipy.integrate import quad
from scipy.stats import norm
from scipy.optimize import Bounds
from scipy.optimize import minimize


class GP_Kernel:
    #working!
    '''
    Contains different possible choices for GP kernels. These are to be used for the correlation matrix of [1].
    '''
    def __init__(self,x1=A([0,0]),x2=A([1,0]),theta=[1,1,1,1,1]): 
        '''
        x1,x2: [longitude,latitude] of locations 1 and 2 to get the correlation for.
        ''' 
        self.x1=x1
        self.x2=x2
        self.theta=theta               

    def W_Noise(self):
        '''
        sig: The variance of the noise is sigma**2.
        x1,x2: [longitude,latitude] of locations 1 and 2 to get the correlation for.
        '''
        sig=self.theta[0]
        if (self.x1[0]==self.x2[0]) and (self.x1[1]==self.x2[1]): 
            return sig**2
        else:
            return 0
        
    def Exp_Quad(self):
        '''
        sig: The variance for a single obs is sigma**2.
        l: lenghtscale
        x1,x2: [longitude,latitude] of locations 1 and 2 to get the correlation for.
        '''
        sig=self.theta[0]
        l=self.theta[1]
        h = np.linalg.norm(self.x1-self.x2)
        return (sig**2)*np.exp( (-h)/(2*(l**2))  )  

    def Rat_Quad(self):
        '''
        sig: The variance for a single obs is sigma**2.
        l: lenghtscale
        a: scale mixture (a>0)
        x1,x2: [longitude,latitude] of locations 1 and 2 to get the correlation for.
        '''
        sig=self.theta[0]
        l=self.theta[1]
        a=self.theta[2]
        if a<0:
            warnings.warn('Rat_Quad parameter a is not >0.')
        h = np.linalg.norm(self.x1-self.x2)
        return (sig**2)*(  ( 1+(  ( (h)/(2*a*(l**2))  )  ) )**(-a)  )
 


class Vcop:

    def __init__(self,X_=mvn.rvs(mean=np.zeros(4)  ,cov=np.identity(4)),obs=[1.23,4.56],n=2,gp_kernel='W_Noise',theta=[1,1,1,1,1],grid=[A([0,1]),A([1,1])]):                 
        '''
        obs: observations 
        n: number of obs
        gp_kernel: kernel to be used
        theta: list of parameters used in correlation functions(sigma,l,aplha) and for the Vcopula(k,m).[sigma in R,l in R,alpha>0,k>0,m in R]. 
        grid: a list of array([longitude.latitude]) for each location.
        '''
        self.z_=[]      #rain values seperated into 0 and non zero.
        self.rain0_1=[] #list of 0-1s to encode 0 and positive rain respectively.
        self.nd=0       #number of obs below the 0-cutoff value(here 0.4 cutoff)
        self.nz=0       #number of obs above the 0-cutoff value. note:nd+nz=n
        for i in obs:
            if i<0.4:
                self.z_.append(0)
                self.rain0_1.append(0)
            else:
                self.z_.append(i)
                self.rain0_1.append(1)
        if obs==[1.23,4.56]:
            self.rain0_1=[1,1,0,1]
        self.n=n
        self.gp_kernel=gp_kernel
        self.theta=theta
        self.grid=grid
        self.X_=X_

    def cor_f(self,loc1,loc2): 
        #working
        '''
        Correlation function.
        loc1,loc2: locations for first and second variable.
        '''   
        if self.gp_kernel=='W_Noise':
            return GP_Kernel(loc1,loc2,self.theta).W_Noise()
        elif self.gp_kernel=='Exp_Quad':
            return GP_Kernel(loc1,loc2,self.theta).Exp_Quad()
        else:
            return GP_Kernel(loc1,loc2,self.theta).Rat_Quad()

    def obs_to_norm(self,z_='none'):
        #TO DO: add the assumed pdf of rainfall for a location. ~glms etc
        '''
        Takes rainfall observations and transforms them to standart normal variables.
        x: a rainfall observation value
        theta: list of parameters used in correlation functions(sigma,l,aplha) and for the Vcopula(k,m).[sigma,l,alpha,k,m]
        '''
        if z_=='none':
            z_=self.z_
        G_z=[]
        for i in z_:
            if i==0:
                G_z.append("GLM PDF 0")
            else:
                G_z.append("GLM PDF non-0")
        y_=[]
        for i in G_z:
            y_.append(norm.ppf(i))
        return y_
  
    def y_to_x(self,y_):
        #working
        #print(Vcop().y_to_x(mvn.rvs(mean=np.ones(2)  ,cov=  np.identity(2))))
        '''
        A V-transform function taking the y variables~N(0,1) to a transformed version x=[x1,x2,...,xn].
        y_: N(0,1) vector [y1,y2,...,yn]
        k: positive scaling constant
        m: arbitrary real number, the cutoff point
        '''
        out=[]
        k=self.theta[3]
        m=self.theta[4]
        for val in y_:
            if val<m:
                out.append(m-val)
            else:
                out.append(k*(val-m))
        return out 

    def h2(self,x1,x2,x_):
        #working
        #print(Vcop(obs=[],n=2,gp_kernel='W_Noise',theta=[1,1,1,1,0],grid=[A([0,1]),A([1,1])]).h2(0,1,Vcop().y_to_x(A([0,0]))))
        '''
        Bivariate PDF for 2 v-transformed variables above the cutoff(ie non-zero).
        x1:index of first variable
        x2:index of second variable
        x_:vector of v-transformed variables
        '''
        k=self.theta[3]
        m=self.theta[4]
        v1=x_[x1]
        v2=x_[x2]
        corr=self.cor_f(self.grid[x1],self.grid[x2])
        corr_mat=A([[1,corr],[corr,1]])
        var=mvn(mean=[0,0],cov=corr_mat)
        out=(1/(k**2))*var.pdf([(v1/k)+m,(v2/k)+m]) + var.pdf([m-v1,m-v2]) + (1/k)*var.pdf([(v1/k)+m,m-v2]) + (1/k)*var.pdf([m-v1,(v2/k)+m])
        return out

    def hc_int(self,z,x1,x2,x_):
        #working
        '''
        Bivariate PDF for 2 v-transformed variables above the cutoff(ie non-zero), but a version used for integrating for the function hc.
        z: variable to integrate over
        x1:index of first variable
        x2:index of second variable
        x_:vector of v-transformed variables
        '''
        k=self.theta[3]
        m=self.theta[4]
        v1=z
        v2=x_[x2]
        corr=self.cor_f(self.grid[x1],self.grid[x2])
        corr_mat=A([[1,corr],[corr,1]])
        var=mvn(mean=[0,0],cov=corr_mat)
        out=(1/(k**2))*var.pdf([(v1/k)+m,(v2/k)+m]) + var.pdf([m-v1,m-v2]) + (1/k)*var.pdf([(v1/k)+m,m-v2]) + (1/k)*var.pdf([m-v1,(v2/k)+m])
        return out

    def hc(self,up_lim,x1,x2,x_):
        #working, but only up to up_lim=35
        #print(Vcop().hc(35,0,1,A([0,0])))
        '''
        Bivariate PDF for 2 V-transformed variables one of which is above the other below the cutoff(1 nonzero and 1 zero). Performs an integral over h2, integrand is the term corresponding to x1.
        up_lim: vtransformed value corresponding to the zero obs. The integral will be over -inf,u     
        x1:index of first v-transformed variable, index of the integrand - 0
        x1:index of second v-transformed variable - non-0
        x_:array of v-transformed variables
        '''
        return quad(self.hc_int,-np.inf,up_lim,args=(x1,x2,x_))[0]

    def h1(self,x):
        '''
        One dimensional marginal density for transformed variables Y_to_X. Is the same for all X.
        x: v-transformed variable, value in R+.
        k: same as the ones used in y_to_x, scaling parameter, k>0
        m: same as the ones used in y_to_x, changing point, in R
        '''
        k=self.theta[3]
        m=self.theta[4]
        return  (1/k)*scs.norm.pdf((x/k)+m)+scs.norm.pdf(m-x)

    def H2(self,x1,x2,x_):
        #working, probably
        #print(Vcop().H2(0,1,A([3,4])))
        '''
        Bivariate CDF for 2 v-transformed variables below the cutoff.
        x1:index of first variable
        x2:index of second variable
        x_:vector of v-transformed variables, have to be >0
        '''
        k=self.theta[3]
        m=self.theta[4]
        v1=x_[x1]
        v2=x_[x2]
        corr=self.cor_f(self.grid[x1],self.grid[x2])
        corr_mat=A([[1,corr],[corr,1]])
        var=mvn(mean=A([0,0]),cov=corr_mat)
        out=var.cdf([(v1/k)+m,(v2/k)+m]) + var.cdf([m-v1,m-v2]) - var.cdf([(v1/k)+m,m-v2]) - var.cdf([m-v1,(v2/k)+m])
        return out
        
    def NLL(self,theta,rain_obs=[0,0]):
        #working
        #test_data=mvn.rvs(mean=np.ones(2)  ,cov= np.identity(2))
        #print(Vcop().NLL(theta=[1,1,1,1,0],rain_obs=Vcop().y_to_x(test_data)))
        '''
        Negative log likelihood. Three parts, one for each combination of 0-nonzero values.
        theta: parameters [sig,l,a,k,m].
        z_: array of rain values, separated into 0-nonzero.
        rain_obs: if given, should be vtransformed variables ready to be used in dists etc. If not given, will transform rain data in Vcop into ready rvs.
        '''
        #initialise
        nll=0  
        if rain_obs==[0,0]:                  #transforms rain data given in Vcop into ready vtransformed variables.
            #z=self.z
            #rain_obs1=Vcop().obs_to_norm()
            #rain_obs=Vcop().y_to_x(rain_obs1) #these are v transformed variables, ready for use in the dists etc.
            rain_obs=self.X_
        sig=theta[0]
        l=theta[1]
        a=theta[2]
        k=theta[3]
        m=theta[4]
        #main
        for couple in combinations(range(self.n),2):
            i=couple[0]
            j=couple[1]       
            #case when both are non-0
            if self.rain0_1[i]==1 and self.rain0_1[j]==1:
                nll+=np.log(self.h2(i,j,rain_obs))
            #case when one is 0 and other 1, but not both 0.
            elif self.rain0_1[i]==0 ^ self.rain0_1[j]==0:
                #identify which one is 0.
                if self.rain0_1[i]==0:
                    nll+=np.log(self.hc(rain_obs[i],i,j,rain_obs))+np.log(self.h1(rain_obs[j]))
                else:
                    nll+=np.log(self.hc(rain_obs[j],j,i,rain_obs))+np.log(self.h1(rain_obs[i]))
                    
            #case when both are 0
            else:
                nll+=np.log(self.H2(i,j,rain_obs))
        return -nll

    def MLE(self):
        '''
        Fucntion to minimize the NLL. Finds theta=[sig,l,a,k,m] most appropriate for the data.
        rain_obs: data->cutoff->N(0,1)->Vtransformed data ready to be used in densities etc.
        '''
        x0=A([20,1,1,1,0]) #initial theta vector

        bounds = Bounds([-np.inf,-np.inf, 0,0,-np.inf],[np.inf,np.inf,np.inf,np.inf,np.inf]) #bounds on theta parameters as [-sig,-l,-a,-k,-m],[+sig,+l,+a,+k,+m]
        res = minimize(Vcop().NLL, x0, method='trust-constr', options={'verbose': 1}, bounds=bounds)
        return res.x

    def Simu(self,theta):
        '''
        Function used for MLE on simulated vtransformed normal data.
        '''
        
        return Vcop().NLL(self,theta)



sim_data=Vcop().y_to_x(mvn.rvs(mean=np.zeros(4)  ,cov=np.identity(4)))
simulation=Vcop(X_=sim_data,n=4,grid=A([[1,0],[1,1],[0,0],[0,1]]))

x0=A([30,3,1,1,2]) #initial theta vector
bounds = Bounds([-np.inf,-np.inf, 0,0,-np.inf],[np.inf,np.inf,np.inf,np.inf,np.inf]) #bounds on theta parameters as [-sig,-l,-a,-k,-m],[+sig,+l,+a,+k,+m]
objective=simulation.NLL
res = minimize(objective, x0,args=simulation.X_, method='trust-constr', options={'verbose': 1}, bounds=bounds)
print(res.x)
'''data=mvn.rvs(mean=np.zeros(2)  ,cov= A([[1,0],[0,1]]))
vdata=Vcop().y_to_x(data)
print(Vcop().NLL([1,1,1,1,0],vdata))'''