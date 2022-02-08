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
from scipy.optimize import Bounds
from scipy.optimize import minimize
import scipy.optimize

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

    def cdf1_x(self,x,k,m):
        '''
        One dimensional cumulative distribution function for transformed variables Y_to_X. Is the same for all X.
        x: value in R to be used for fX(x)
        k: same as the ones used in y_to_x, scaling parameter, k>0
        m: same as the ones used in y_to_x, changing point, in R
        '''
        return scs.norm.cdf((x/k)+m)-scs.norm.cdf(m-x)

    def pdf1_x(self,x,k,m):
        '''
        One dimensional marginal density for transformed variables Y_to_X. Is the same for all X.
        x: value in R to be used for fX(x)
        k: same as the ones used in y_to_x, scaling parameter, k>0
        m: same as the ones used in y_to_x, changing point, in R
        '''
        return  (1/k)*scs.norm.pdf((x/k)+m)+scs.norm.pdf(m-x)

    def b_x(self,x,k):
        '''
        A function used inside the zeta function, which is needed for the copula.
        x: input value
        k:scaling parameter, k>0
        '''
        if x>0:
            return 1/k
        else:
            return -1

    def zeta(self,x_,k,d_):
        '''
        a Function used in the copula expression.
        x_: vector of n transformed values
        k: scaling parameter, k>0. Needed for b_x function here.
        d_: a vector of 1s and 0s related to the x_ vector.
        '''
        out=[]
        for val,pow in zip(x_,d_):
            out.append([val*self.b_x((-1)**(pow),k)])
        return out

    def pdfn_x(self,x_,k,m,cor_mat):
        '''
        function to get the joint density function for [x1,x2,...xn], x1:n being the thransformed rv/obs Y.
        x_: vector of n transformed values
        k:  same as the ones used in y_to_x, scaling parameter, k>0. Needed for b_x function here.
        m: same as the ones used in y_to_x, changing point, in R
        cor_mat: the correlation matrix for the observation vector y1:n
        '''
        #i=0 # a counting variable to assess progress when running the code
        n=len(x_)
        m_=m*np.ones((n,1))
        out=1/(   ( (2*pi)**(n/2) )*sqrt(D(I(cor_mat)))   ) #initialising the joint density
        for d in list(product(range(2),repeat=n)):
            #i+=1 #to assess progress
            #if i%20==0:
                #print(100*i/(2**n))

            S=0 # initialising the smaller sum inside the expression. S is the sum of elements in d.
            for elem in d:
                S+=elem
            out+=(   1/( k**(n-S) )   ) *  exp(-0.5*   T(self.zeta(x_,k,d_=A(d))+m_) @ I(cor_mat) @  (self.zeta(x_,k,d_=A(d))+m_)  )
        return out

    def vnor_cop(self,x_,k,m,cor_mat):
        '''
        V-transformed multivariate normal copula containing dependencies between locations.
        x_: vector n Vtransformed variables
        k:  same as the ones used in y_to_x, scaling parameter, k>0. Needed for b_x function here.
        m: same as the ones used in y_to_x, changing point, in R
        cor_mat: the correlation matrix for the observation vector y1:n
        '''
        denom=1
        for val in x_:
            denom=denom*self.pdf1_x(val,k,m)
        return self.pdfn_x(x_,k,m,cor_mat)/denom


    def simulate(self,theta,sets):
        #working
        #Vcop08(3,A([[0,0],[1,0],[0,1]]),'Exp_Quad').simulate([1,1,1,1],4)
        '''
        Function to simulate m sets of uniform [0,1] realisatoins for each location.
        theta: parameter vector [l,alpha,k,m] where l,alpha are used in the correlation matrix, and k,m in the vtransform.
        sets: natural number corresponding to the number of simulated sets of n realisations.
        returns: Z=m sets of n Vtransformed variables, and U=m sets of n uniform[0,1] vtransformed variables. 
        output: simulate[0] is the Z=Vtransforms, simulate[1]=U=uniformed Vtransforms.
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

    def NegativeLogLikelihood(self,theta,V_data='none',U_data='none'):
        #works for V_data - still need to do U_data.
        #test=Vcop08(3,A([[0,0],[1,0],[0,1]]),gpk='Exp_Quad')
        #data=test.simulate([1,1,1,1],4)[0]
        #estim=test.LogLikelihood([1,1,1,1],V_data=data)
        '''
        A functoin to evaluate the log-likelihood for a given set of data.
        theta: parameter vector [l,alpha,k,m] where l,alpha are used in the correlation matrix, and k,m in the vtransform.
        V_data: Vtransformed variables. If not given, evaluation will be done using U_data instead.
        U_data: uniform[0,1] Vtransformed variables.If not given, evaluation will be done using V_data instead.
        '''
        ll=0
        if U_data=='none':#This is for Simulated Vtransformed variables
            sets=int(len(V_data))
            #Construct Sigma - the correlation matrix
            n=self.n
            m=theta[3]
            k=theta[2]
            loc=self.locations
            sigma=np.identity(n)
            for couple in combinations(range(n),2):
                i=couple[0]
                j=couple[1]
                cor=self.cor_f(loc[i],loc[j],theta)
                sigma[i][j]=cor
                sigma[j][i]=cor
            
            #Compute ll= summ over m of log(cop_sendity of set m)
            for s in range(sets):
                x_=V_data[s]
                ll+=np.log(self.vnor_cop(x_,k,m,sigma))
                      
            return -ll

        else:#This is for U[0,1] observed marginal cdfs
            m=len(V_data)
            print('U')
        
        
test=Vcop08(6,A([[0,0],[0,1],[1,0],[1,1],[0,4],[4,0]]),gpk='Exp_Quad')        
#To simulate data:
Z_data=test.simulate(theta=[2,2,2,2],sets=50)[0]

#To estimate parameters:
x0=A([1,1,1,1]) #initial theta vector
bounds = Bounds([-np.inf, 0,0,-np.inf],[np.inf,np.inf,np.inf,np.inf]) #bounds on theta parameters as [-l,-a,-k,-m],[+l,+a,+k,+m]

est = minimize(test.NegativeLogLikelihood, x0,args=Z_data, method='trust-constr', options={'verbose': 1,'maxiter':800000}, bounds=bounds)
print(est.x)

