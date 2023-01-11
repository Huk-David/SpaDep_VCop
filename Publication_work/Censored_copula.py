import numpy as np
import scipy.stats as scs
from sklearn.gaussian_process.kernels import Matern
from tqdm import tqdm
from sklearn.metrics.pairwise import rbf_kernel as rbf

def mvn_cdf(a,b,Sigma):
    m = len(a)
    # Reorder Sigma, a and b according to the length of interval (higher earlier)
    len_interval = np.zeros(m)
    for ind in range(m):
        if a[ind] == -np.inf:
            a[ind] = -100000
        if b[ind] == np.inf:
            b[ind] = 100000
        len_interval[ind] = b[ind] - a[ind]
    order_len_interval = [int(i[0]) for i in sorted(enumerate(len_interval), key=lambda x:x[1], reverse=True)]
    Sigma = Sigma[order_len_interval, :][:,order_len_interval]
    a = [a[x] for x in order_len_interval]
    b = [b[x] for x in order_len_interval]


    d = np.zeros(shape=(m,))
    e = np.ones(shape=(m,))
    f = np.zeros(shape=(m,))
    y = np.zeros(shape=(m,))
    # these can be changed for different precision levels
    epsilon = 1.e-5
    alpha = 2.5
    N_max = int(1e5)

    # Compute Cholesky decomposition of Sigma to produce lower trinagualr matrix
    C = np.linalg.cholesky(Sigma)

    #Sigma_sparse = sparse.csc_matrix(Sigma)
    #C_sparse = cholmod.cholesky(Sigma_sparse).L()
    #C = C_sparse

    Intsum, N, Varsum = 0, 0, 0

    if a[0] != -np.inf:
        d[0] = scs.norm.cdf(a[0]/C[0,0])
    if b[0] != np.inf:
        e[0] = scs.norm.cdf(b[0]/C[0,0])
    f[0] = np.log(e[0] - d[0])

    for ind_N in range(N_max):
        
        w = np.random.rand(m)
        for ind_m in range(1,m):
            y[ind_m-1] = scs.norm.ppf(d[ind_m-1]+w[ind_m-1]*(e[ind_m-1]-d[ind_m-1]))
            if a[ind_m] != -np.inf:
                d[ind_m] = scs.norm.cdf((a[ind_m]-sum(C[ind_m, :ind_m]*y[:ind_m]))/C[ind_m,ind_m])
            if b[ind_m] != np.inf:
                e[ind_m] = scs.norm.cdf((b[ind_m]-sum(C[ind_m, :ind_m]*y[:ind_m]))/C[ind_m,ind_m])
            f[ind_m] = np.log((e[ind_m] - d[ind_m])) + f[ind_m-1]
        N = N+1
        delta = (f[-1]-Intsum)/N
        Intsum = Intsum + delta
        Varsum = (N-2)*Varsum/N + pow(delta, 2)
        Error = alpha*np.sqrt(Varsum)
        if Error < epsilon:
            break

    return(Intsum)

# Truncated Gaussian copula
class truncgauss():
    def __init__(self,L):
        '''
        Class to fit and work with a truncated Gaussian copula, using two functions.
        sim: simulated [0,1] values using this copula density.
        eval_nll: evaluate the negative log likelihood based on given data.
        To use the class, the dimension of the data is needed.
        n: dimension of data
        '''
        self.L=L
    
    def sim(self,theta,inv_us,draws,day_idx,kernel='rbf'):
        '''
        Simulates m draws from the truncated gaussian copula conditional on parameters theta for the covariance kernel.
        '''
        # Generate mvn with Sigma, then truncate according to p_i
        n = len(self.L)
        if kernel =='rbf':
            cov_mat = rbf(self.L,gamma=theta)
        elif kernel == 'Matern':
            cov_mat = Matern(length_scale=theta[0],nu=theta[1]).__call__(self.L)
        rvs = scs.multivariate_normal.rvs(np.zeros(n),cov_mat,size=draws)

        return [[ scs.norm.cdf(max([rvs[i][j],-inv_us[day_idx][j]])) for j in range(n)] for i in range(draws)]

    def nll_Matern(self,theta,Invcdf_Us,truncation_pi,did_rain,len_locs):
            cov_mat = np.nan_to_num(Matern(length_scale=theta[0],nu=theta[1]).__call__(self.L))
            nll=0
            for day in tqdm(range(len(Invcdf_Us))): #eg [251,...,500]
                
                # numerator: joint pdf integrated over truncation - equivalent to elegant normal cdf in some cases. Check cases:
                if np.sum(did_rain[day])==len(did_rain[day]): #all wet, just normal pdf
                    nll += scs.multivariate_normal.logpdf(x=Invcdf_Us[day],mean=np.zeros(len(Invcdf_Us[0])),cov=cov_mat)
                else: # some dry -> use elegant cdf
                    #print('elegant'+str(np.sum(did_rain[day])))
                    if np.sum(did_rain[day])==0: # all dry -> use normal cdf
                        nll += scs.multivariate_normal.logcdf(x=Invcdf_Us[day],mean=np.zeros(len(Invcdf_Us[0])),cov=cov_mat)

                    else:
                        
                        # need to re-arrange the covariance matrix into 2*2 blocks based on censored/uncensored. 
                        r_rain,c_rain = [[i] for i in np.nonzero(did_rain[day])[0]],[i for i in np.nonzero(did_rain[day])[0]]
                        r_dry,c_dry = [[i] for i in range(len_locs) if i not in np.nonzero(did_rain[day])[0]],[i for i in range(len_locs) if i not in np.nonzero(did_rain[day])[0]]
                        
                        # create submatricies
                        submat11 = cov_mat[r_dry,c_dry]
                        submat12 = cov_mat[r_dry,c_rain]
                        submat21 = cov_mat[r_rain,c_dry]
                        submat22 = cov_mat[r_rain,c_rain]
                        
                        #### New estimate part      ####

                        if len(submat22)!=1:
                            sub_cov = (submat11-submat12@np.linalg.inv(submat22)@submat21)#.reshape(len(c_dry),len(c_dry))
                        else: # submat22 is an integer, aka only one non-zero obs
                            sub_cov = (submat11-submat12*np.linalg.inv(submat22)[0]*submat21)#.reshape(len(c_dry),len(c_dry))
                        
                        # Do new approximation to cdf
                        test = [c_dry,[Invcdf_Us[day][k] for k in c_dry],sub_cov]
                        nll += mvn_cdf([-np.inf for i in range(len(c_dry))],[Invcdf_Us[day][k] for k in c_dry],sub_cov)                               

                        #### New estimate part - end ####

                # denominator: multiplication of marginals - pdf for positive rain, cdf for truncated
                for loc in range(len(Invcdf_Us[0])):
                    if did_rain[day][loc]==1:# wet: add pdf(invcdf(u_i))
                        nll -= scs.norm.logpdf(Invcdf_Us[day][loc],loc=0,scale=1)
                    else:# dry: add cdf(invcdf(pi))
                        nll -= np.log(truncation_pi[day][loc])
                
            return -nll
