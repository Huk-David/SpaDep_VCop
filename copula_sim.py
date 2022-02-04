#will simulate a mvrn dist with 10 obs and try copula downscaling on it

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


#[[1],[4]] is a column
#[[1,4]] is a row
#@ to multiply matricies/vectors





def y_to_x(y_,k,m):
    '''
    A function taking the y variables/observations to a transformed version x.
    y_: observation vector [y1,y2,...,yn]
    k: positive scaling constant
    m: arbitrary real number, the cutoff point
    '''
    out=[]
    for obs in y_:
        if obs<m:
            out.append(m-obs)
        else:
            out.append(k*(obs-m))
    return out

def cdf1_x(x,k,m):
    '''
    One dimensional cumulative distribution function for transformed variables Y_to_X. Is the same for all X.
    x: value in R to be used for fX(x)
    k: same as the ones used in y_to_x, scaling parameter, k>0
    m: same as the ones used in y_to_x, changing point, in R
    '''
    return scs.norm.cdf((x/k)+m)-scs.norm.cdf(m-x)

def pdf1_x(x,k,m):
    '''
    One dimensional marginal density for transformed variables Y_to_X. Is the same for all X.
    x: value in R to be used for fX(x)
    k: same as the ones used in y_to_x, scaling parameter, k>0
    m: same as the ones used in y_to_x, changing point, in R
    '''
    return  (1/k)*scs.norm.pdf((x/k)+m)+scs.norm.pdf(m-x)

def b_x(x,k):
    '''
    A function used inside the zeta function, which is needed for the copula.
    x: input value
    k: same as the ones used in y_to_x, scaling parameter, k>0
    '''
    if x>0:
        return 1/k
    else:
        return -1

def zeta(x_,k,d_):
    '''
    a Function used in the copula expression.
    x_: vector of n transformed values
    k:  same as the ones used in y_to_x, scaling parameter, k>0. Needed for b_x function here.
    d_: a vector of 1s and 0s related to the x_ vector.
    '''
    out=[]
    for val,pow in zip(x_,d_):
        out.append([val*b_x((-1)**(pow),k)])
    return out

def pdfn_x(x_,k,m,cor_mat):
    '''
    function to get the joint density function for [x1,x2,...xn], x1:n being the thransformed rv/obs Y.
    x_: vector of n transformed values
    k:  same as the ones used in y_to_x, scaling parameter, k>0. Needed for b_x function here.
    m: same as the ones used in y_to_x, changing point, in R
    d_: a vector of 1s and 0s corresponding to the x_ vector.
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
        out+=(   1/( k**(n-S) )   ) *  exp(-0.5*   T(zeta(x_,k,d_=A(d))+m_) @ I(cor_mat) @  (zeta(x_,k,d_=A(d))+m_)  )
    return out

def vnor_cop(y_,k,m,cor_mat):
    '''
    V-transformed multivariate normal copula containing dependencies between locations.
    y_: vector n values
    k:  same as the ones used in y_to_x, scaling parameter, k>0. Needed for b_x function here.
    m: same as the ones used in y_to_x, changing point, in R
    cor_mat: the correlation matrix for the observation vector y1:n
    '''
    x_=y_to_x(y_=y_,k=k,m=m)
    denom=1
    for val in x_:
        denom=denom*pdf1_x(val,k,m)
    return pdfn_x(x_,k,m,cor_mat)/denom

def exp_cor(loc1,loc2,A,B):
    '''
    loc1: location 1 used to compute the distance h 
    loc2: location 2 used to compute the distance h 
    A,B: parameters of exponential correlation function
    '''
    if loc1==loc2:
        return 1
    else:
        h=1# have to modify to represent distances between locations
        return B*np.exp(-(h/A))


'''
#------------- to get one run for n variables. Returns a single vnorm copula density value-----------



cov_mat=np.identity(n)
k=1
m=1.5
m_=m*np.ones((n,1))

print(vnor_cop(y_=y_,k=k,m=m,cor_mat=cov_mat))

'''
# Optimization methods with scipy.optimize

'''#1 Nelder-Mead Simplex algorithm (method='Nelder-Mead') - working, not appropriate
#Not sure how to also optimize matrix. Also does not take constraonts for k -> discarded

from scipy.optimize import minimize
def test_cop(vector):
    K=vector[0]
    M=vector[1]
    return vnor_cop(y_=y_,k=K,m=M,cor_mat=cov_mat)
x0=[1,-4]
res=minimize(test_cop, x0, method='nelder-mead', options={'xatol': 0.01, 'disp': True,'maxfev':1000})
print(res.x)'''

#2Trust-Region Constrained Algorithm (method='trust-constr)
from scipy.optimize import Bounds
from scipy.optimize import minimize

#simulated data- will be replaced by the model marginals
#generating 1 measurements of n dimensional mvn variables 
n=3
y_=(mvn.rvs(mean=np.ones(n)  ,cov=  np.identity(n)))
cov_mat=np.identity(n)
k=2
m=1.5
m_=m*np.ones((n,1))

#specifications for minimize()
bounds = Bounds([0, -np.inf,0,0],[np.inf,np.inf,np.inf,1])# [Left bounds 1 to 4],[right bounds 1 to 4]: first for k>0,then m in R, then for A>0 and 1>B>0, two parameters of the exponential correlation function [Bardossy 2011]

def test_cop(vector):
    K=vector[0]
    M=vector[1]
    A=vector[2]
    B=vector[3]
    for i in range(n):
        for j in range(n):
            cov_mat[i][j]=exp_cor(i,j,A=A,B=B)
    return -vnor_cop(y_=y_,k=K,m=M,cor_mat=cov_mat)

x0=[1,2,2,0.5] #starting values for k, m, A, B

res = minimize(test_cop, x0, method='trust-constr', options={'verbose': 1}, bounds=bounds)
print(res.x)

