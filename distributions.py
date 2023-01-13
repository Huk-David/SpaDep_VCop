from torch.distributions import bernoulli
from torch.distributions.distribution import Distribution
from torch.distributions.gamma import Gamma
from torch.distributions.half_normal import HalfNormal
from torch.distributions.log_normal import LogNormal
from torch.distributions.normal import Normal
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.poisson import Poisson

from torch.distributions.transformed_distribution import TransformedDistribution
from typing import Callable, Optional, Union
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler, FunctionTransformer, MaxAbsScaler

from torch.distributions import constraints
import torch
import numpy as np
from torch import Tensor


# Distributions
class LogNormalHurdle():
    r"""
    Creates a log-normal distribution parameterized by
    :attr:`mu` and :attr:`scale` where::

        X ~ Normal(mu, scale)
        Y = exp(X) ~ LogNormal(mu, scale)

    Example::

        >>> m = LogNormal(torch.tensor([0.0]), torch.tensor([1.0]))
        >>> m.sample()  # log-normal distributed with mean=0 and stddev=1
        tensor([ 0.1046])

    Args:
        mu (float or Tensor): mean of log of distribution
        scale (float or Tensor): standard deviation of log of the distribution
    """
    arg_constraints = {'mu': constraints.real, 'scale': constraints.positive}
    support = constraints.positive
    has_rsample = True

    def __init__(self,mu=0.0, disp=1.0, prob=0.5, validate_args=None):
        mu = torch.as_tensor(mu)
        disp = torch.as_tensor(disp) 

        self.set_parameters(mu, disp, prob, validate_args)

    def set_parameters(self, mu, disp, prob, validate_args=None):
        
        if not hasattr(self, 'bernoulli_dist') or not self.bernoulli_dist.probs.equal( prob).all():
            self.bernoulli_dist = Bernoulli(prob, validate_args=validate_args)
        
        if not hasattr(self, 'lognormal_dist') or not self.lognormal_dist.base_dist.loc.equal( torch.log(mu) ).all() or not self.lognormal_dist.base_dist.scale.equal( disp.pow(0.5) ).all() :
            self.lognormal_dist = LogNormal( torch.log(mu) , disp.pow(0.5), validate_args=validate_args)

    def sample(self, sample_size=(1,) ):
        rain_prob = self.bernoulli_dist.sample( sample_size )

        sampled_rain = torch.where(rain_prob>=0.5, self.lognormal_dist.sample( (1,) ), 0  )

        return sampled_rain

    def unscale_distribution( self, mu,  disp,  p, scaler ):
        """When standardizing/destandardizing a glm's output we must be concious of the distribution we are sampling from

            Here we provide logic that provides distribution specific scaling for the mu and dispersion terms we predict

            NOTE: the destandardization we use is for the Exponential Dispersion versions of Gamma

        Args:
            mu (Union[Tensor, np.ndarray]): [description]
            disp ([type]): [description]
            target_distribution_name ([type]): [description]
            scaler (Union[MinMaxScaler,StandardScaler,]): [description]

        Returns:
            [type]: [description]
        """
        
        # mu = mu + torch.log( mu.new_tensor( 1/scaler.scale_) )
        mu = mu + torch.log( mu.new_tensor( scaler.scale_) )
        
        disp = disp

        return mu, disp, p

    @classmethod    
    def get_mean(cls, mu, disp, p):
        if isinstance(mu, torch.Tensor):
            mean =  torch.where( p>=0.5, torch.exp( mu + disp.pow(2)/2), mu.new_tensor(0.0) )
        elif isinstance(mu, np.ndarray):
            mean = np.where( p>=0.5, np.exp( mu + np.power(disp,2)/2), 0.0 )

        return mean
    @classmethod
    def get_variance(cls, mu, disp, p):
        if isinstance(mu, torch.Tensor):
            var =  torch.where( p>=0.5, ( torch.exp(disp) - 1 ) * torch.exp(2*mu + disp), mu.new_tensor(0.0) )
        elif isinstance(mu, np.ndarray):
            var =  np.where( p>=0.5, ( np.exp(disp) - 1 ) * np.exp(2*mu + disp), 0.0 )

        return var 
    
    @classmethod
    def get_mode(cls, mu, disp, p):
        if isinstance(mu, torch.Tensor):
            mode =  torch.where( p>=0.5, torch.exp(mu-disp), mu.new_tensor(0.0))
        elif isinstance(mu, np.ndarray):
            mode =  np.where( p>=0.5, np.exp(mu-disp), 0.0 )
        return mode

    @classmethod
    def get_skewness(self, mu, disp, p):
        if isinstance(mu, torch.Tensor):
            skewness =  torch.where( p>=0.5, (torch.exp(disp)+2)*(torch.exp(disp)-1).pow(0.5), mu.new_tensor(0.0) )
        elif isinstance(mu, np.ndarray):
            skewness =  np.where( p>=0.5, (np.exp(disp)+2)*(np.power( np.exp(disp)-1), 0.5), 0.0 )

        return skewness

class GammaHurdle():
    r"""
    Creates a Gamma Hurdle distribution parameterized by
    :attr:`loc` and :attr:`scale` where and :attr:`p` :

        p ~ Bernoulli( p )
        Y ~ Gamma( mu, disp) ~

    
    """
    arg_constraints = {'mu': constraints.positive, 'disp': constraints.positive, 'prob':constraints.positive}
    support = constraints.positive
    has_rsample = True

    def __init__(self, mu=2.0, disp=0.2, prob=0.5, validate_args=None):
        
        mu = torch.as_tensor(mu)
        disp = torch.as_tensor(disp)

        self.set_parameters(mu, disp, prob, validate_args)

    def set_parameters(self, mu, disp, prob, validate_args=None):
        if not hasattr(self, 'bernoulli_dist') or not self.bernoulli_dist.probs.equal( prob).all():
            self.bernoulli_dist = Bernoulli(prob, validate_args=validate_args)
        
        alpha, beta, prob = self.reparameterize( mu, disp, prob )

        if not hasattr(self, 'gamma_dist') or not self.gamma_dist.base_dist.loc.equal( alpha ).all() or not self.lognormal_dist.base_dist.scale.equal( beta ).all() :
            self.gamma_dist = Gamma(alpha, beta, validate_args=validate_args)
            
    def sample(self,sample_size=(1,)):
        
        rain_prob = self.bernoulli_dist.sample( sample_size )

        sampled_rain = torch.where(rain_prob>=0.5, self.gamma_dist.sample( (1,) ), 0  )

        return sampled_rain

    def reparameterize(self, mu, disp, prob):
        # Converts from the \mu, \sigma^2 parameterization to the \alpha , \beta parameterization

        alpha = 1/disp
        beta = alpha/mu

        return alpha, beta, prob

    def unscale_distribution( self, mu: Union[Tensor, np.ndarray], disp, 
                            p,
                            scaler: Union[MinMaxScaler] ):
        """When standardizing/destandardizing a glm's output we must be concious of the distribution we are sampling from

            Here we provide logic that provides distribution specific scaling for the mu and dispersion terms we predict

            NOTE: the destandardization we use is for the Exponential Dispersion versions of Gamma

        Args:
            mu (Union[Tensor, np.ndarray]): [description]
            disp ([type]): [description]
            target_distribution_name ([type]): [description]
            scaler (Union[MinMaxScaler,StandardScaler,]): [description]

        Returns:
            [type]: [description]
        """
        
        mu = mu * 1/mu.new_tensor(scaler.scale_)
        disp = disp

        return mu, disp, p

    
    @classmethod
    def get_mean(cls, mu, disp, p):
        if isinstance(mu, torch.Tensor):
            mu = torch.where( p>=0.5, mu, mu.new_tensor( 0.0 )  )

        elif isinstance(mu, np.ndarray):
            mu = np.where( p>=0.5, mu, 0.0)
        return mu

    @classmethod
    def get_variance(cls, mu, disp, p):
        if isinstance(mu, torch.Tensor):
            var = torch.where( p>=0.5, disp * mu.pow(2), 0.0  )
        elif isinstance(mu, np.ndarray):
            var = np.where( p>=0.5, disp * np.power(mu,2), 0.0  )
        return var

    @classmethod
    def get_mode(cls, mu, disp):
        return None
    
    @classmethod
    def get_skewness(self, mu, disp, p):
        if isinstance(mu, torch.Tensor):
            skewness = torch.where( p>0.5, 2 * np.power(disp,0.5), 0.0 )
        return skewness

class CompoundPoisson():

    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.positive
    has_rsample = True

    def __init__(self, mu=1.0, disp=1.0, p=1.5, validate_args=None):

        lambda_, alpha, beta = self.reparameterize( torch.as_tensor(mu), 
                                                    torch.as_tensor(disp),
                                                    torch.as_tensor(p) )
        
        self.poisson_dist = Poisson(lambda_, validate_args=validate_args)
        self.gamma_dist = LogNormal(alpha, beta, validate_args=validate_args)

    def sample(self,sample_size=(1,)):

        N = self.poisson_dist.sample( sample_size )
        li_gammas = [ self.gamma_dist.sample( sample_size ) for i in torch.arange(N)]
        rain = torch.stack( li_gammas, dim=0).sum(dim=0)

        return rain

    def unscale_distribution( self, mu: Union[Tensor, np.ndarray], disp, 
                            p, 
                            scaler: Union[MinMaxScaler] ):
        """When standardizing/destandardizing a glm's output we must be concious of the distribution we are sampling from

            Here we provide logic that provides distribution specific scaling for the mu and dispersion terms we predict

            NOTE: the destandardization we use is for the Exponential Dispersion versions of Gamma

        Args:
            mu (Union[Tensor, np.ndarray]): [description]
            disp ([type]): [description]
            target_distribution_name ([type]): [description]
            scaler (Union[MinMaxScaler,StandardScaler,]): [description]

        Returns:
            [type]: [description]
        """
        # scale_ = torch.as_tensor(scaler.scale_, device=mu.device)
        scale_ = mu.new_tensor(scaler.scale_)
        mu = mu * 1/scale_
        disp = disp *( (1/scale_)**(2-p))
        return mu, disp, p

    def reparameterize(self, mu, disp, p):
        # Convert from ED form to standard form

        lambda_ = mu.pow(2-p) * ( disp*(2-p) ).pow(-1)
        alpha = disp*(p-1)*mu.pow(p-1)
        beta = (2-p)/(p-1)

        return lambda_, alpha, beta
    
    @classmethod
    def get_mean(cls, mu, disp, p):
        if isinstance(mu, torch.Tensor):
            mu = torch.where(mu>1.0, mu, mu.new_tensor(0.0))
        elif isinstance(mu, np.ndarray):
            mu = np.where(mu>1.0, mu, 0.0)
        return mu

    @classmethod
    def get_variance(self, mu, disp, p):
        if isinstance(mu, torch.Tensor):
            var = torch.where( mu>1.0, disp * mu.pow(p), mu.new_tensor(0.0))

        elif isinstance(mu, np.ndarray):
            var = np.where( mu>1.0, disp * np.power(mu, p), 0.0)
        
        return var
    
    @classmethod
    def get_mode(self, mu, disp, p):
        return None

    @classmethod
    def get_skewness(self, mu, disp, p):
        return None

    
    