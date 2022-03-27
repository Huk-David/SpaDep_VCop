from cmath import log
import torch
from torch import distributions
from torch.distributions.gamma import Gamma
from torch.distributions.half_normal import HalfNormal
from torch.distributions.log_normal import LogNormal
from torch.distributions.normal import Normal

import torch.functional as F
from torch.nn.modules.loss import GaussianNLLLoss, PoissonNLLLoss
from scipy import stats

from torch.nn.modules.loss import _Loss
from torch import Tensor
from typing import Callable, Optional
import math
import numpy as np
#TODO: Add to notes: Dropout Inference in Bayesian Neural Networks, page 3, column 2, last paragraph of section 2.3

class LogNormalHurdleNLLLoss(_Loss):
    """A log normal distribution LN(\mu, disp) with \mu,disp \in [0, \inf] 

    """
    __constants__ = ['full', 'eps', 'reduction']
    full: bool
    eps: float

    def __init__(self, *, full: bool = False, eps: float = 1e-6, reduction: str = 'mean', pos_weight=1, **kwargs ) -> None:
        super(LogNormalHurdleNLLLoss, self).__init__(None, None, reduction)
        self.full = full
        self.eps = eps
        self.register_buffer('pos_weight',torch.tensor([pos_weight]))
        self.register_buffer('pi',torch.tensor(math.pi))
        self.bce_logits = torch.nn.BCEWithLogitsLoss(reduction='sum', pos_weight=self.pos_weight)
        assert reduction == 'mean'
    
    def lognormal_nll(self, obs, mu, disp ):

        # disp = disp.clone()
        # obs = obs.clone()
        # with torch.no_grad():
        #     # disp.clamp_(min=self.eps)
        #     # obs.clamp_(min=self.eps)
        #     obs.clamp_(min=mu.exp().min())

        logobs = torch.log(obs)
        logobs = logobs.clone()
        with torch.no_grad():
            logobs.clamp_(min=mu.min())
            
        ll  = -0.5 * (torch.log(disp) + (logobs - mu )**2 / disp) 
                        
        if self.full:
            ll += -0.5*torch.log(2*self.pi) - logobs

        return -ll

    def forward(self, rain: Tensor, did_rain:Tensor, mu: Tensor, disp: Tensor, logits: Tensor, **kwargs) -> Tensor:
        """
            rain : true rain fall
            did_rain: whether or not it rained
            mu: predicted mu
            disp: predicted dispiance
            logits: predicted prob of rain
        """
        if did_rain.dtype != logits.dtype:
            did_rain = did_rain.to(logits.dtype)

        if rain.dtype != mu.dtype:
            rain = rain.to(mu.dtype)

        # Check disp size
        # If disp.size == rain.size, the case is heteroscedastic and no further checks are needed.
        # Otherwise:
        if disp.size() != rain.size():

            # If disp is one dimension short of rain, but the sizes match otherwise, then this is a homoscedastic case.
            # e.g. rain.size = (10, 2, 3), disp.size = (10, 2)
            # -> unsqueeze disp so that disp.shape = (10, 2, 1)
            # this is done so that broadcasting can happen in the loss calculation
            if rain.size()[:-1] == disp.size():
                disp = torch.unsqueeze(disp, -1)

            # This checks if the sizes match up to the final dimension, and the final dimension of disp is of size 1.
            # This is also a homoscedastic case.
            # e.g. rain.size = (10, 2, 3), disp.size = (10, 2, 1)
            elif rain.size()[:-1] == disp.size()[:-1] and disp.size(-1) == 1:  # Heteroscedastic case
                pass

            # If none of the above pass, then the size of disp is incorrect.
            else:
                raise ValueError("disp is of incorrect size")

        # Check validity of reduction mode
        if self.reduction != 'none' and self.reduction != 'mean' and self.reduction != 'sum':
            raise ValueError(self.reduction + " is not valid")

        # logits / etc loss
       
        # Calculate the hurdle burnouilli loss
        loss_norain = self.bce_logits( logits, did_rain)

        # Calculate the ll-loss
        indices_rainydays = torch.where(did_rain.view( (1,-1))==1)
        

        rain_rainydays = rain.view((1,-1))[indices_rainydays]
        mu_rainydays = mu.view((1,-1))[indices_rainydays]
        disp_rainydays = disp.view((1,-1))[indices_rainydays]

        loss_rain = self.lognormal_nll(rain_rainydays, mu_rainydays, disp_rainydays)

        loss_rain = loss_rain.sum()
        loss_rain = self.pos_weight * loss_rain

        if self.reduction == 'mean':
            loss_norain = loss_norain/rain.numel()
            loss_rain = loss_rain/rain.numel()
            loss = loss_norain + loss_rain 

        return loss, {'loss_norain':loss_norain.detach() , 'loss_rain':loss_rain.detach()}            

    def prediction_metrics(self, rain, did_rain, mean, logits, **kwargs):

        #Classification losses
        pred_rain_bool = torch.where( logits>=0.0, 1.0, 0.0)
        pred_acc = torch.mean( pred_rain_bool*did_rain + (pred_rain_bool-1)*(did_rain-1) )
        pred_rec = (did_rain*pred_rain_bool).sum() / did_rain.sum()

        #Continuous losses
        indices_rainydays = torch.where(did_rain.view( (1,-1))>0)
        
        pred_mse = torch.nn.functional.mse_loss(
            mean.view((1,-1))[indices_rainydays],
            rain.view((1,-1))[indices_rainydays]
        ) 
        #MSE on days it did rain
        

        pred_metrics = {'pred_acc': pred_acc,
                        'pred_rec':pred_rec,
                            'pred_mse': pred_mse }
        return pred_metrics

class GammaHurdleNLLLoss(_Loss):
    """Gamma distribution model with an Exponetial Dispersion parameterization.
        To model rainfall y
            We predict value p between 0 and 1  .
            If p<0.5, then y=0
            if p>0.5, then we use a Gamma distribution to sample the rainfall
    """
    __constants__ = ['full', 'eps', 'reduction']
    full: bool #Often extra parameters work to regularise the parameters
    eps: float

    def __init__(self, *, full: bool = True, eps: float = 1e-6, reduction: str = 'mean', pos_weight=1, **kwargs ) -> None:
        super(GammaHurdleNLLLoss, self).__init__(None, None, reduction)
        self.full = full
        self.eps = eps
        self.register_buffer('pos_weight',torch.tensor([pos_weight]) )
        self.bce_logits = torch.nn.BCEWithLogitsLoss(reduction='sum', pos_weight=self.pos_weight)
        
        assert reduction == 'mean'
    
    def gamma_nll(self, obs,#observations
                    mu,     # mu
                    disp):  # dipersion - \alpha^-1 is dispersion term
        # Uses the Gamma(\mu, \sigma^2) parameterization. Where \mu = \frac{\alpha}{\beta} and \sigma^2 = \frac{1}{\alpha}
        # original parameteriszation was Gamma( \alpha, \beta )

        d = disp
        # Clamp for stability
        obs = obs.clone()
        with torch.no_grad():
            obs.clamp_(min=self.eps)

        ll = (d.pow(-1)-1)*torch.log(obs) - obs*mu.pow(-1)*d.pow(-1) - torch.lgamma(d.pow(-1)) - d.pow(-1) * torch.log(d*mu)
                   
        return -ll

    def forward(self, rain: Tensor, did_rain:Tensor, mu: Tensor, disp: Tensor, logits: Tensor, **kwargs) -> Tensor:
        """
            rain : true rain fall
            did_rain: whether or not it rained
            mu: predicted mu
            disp: predicted dispersion - note dispersion term is the inverse of the \alpha value from the traditional gamma distribution. 
            logits: predicted prob of rain
        """
        if did_rain.dtype != logits.dtype:
            did_rain = did_rain.to(logits.dtype)

        if rain.dtype != mu.dtype:
            rain = rain.to(mu.dtype)

        # Check disp size
        # If disp.size == rain.size, the case is heteroscedastic and no further checks are needed.
        # Otherwise:
        if disp.size() != rain.size():

            # If disp is one dimension short of rain, but the sizes match otherwise, then this is a homoscedastic case.
            # e.g. rain.size = (10, 2, 3), disp.size = (10, 2)
            # -> unsqueeze disp so that disp.shape = (10, 2, 1)
            # this is done so that broadcasting can happen in the loss calculation
            if rain.size()[:-1] == disp.size():
                disp = torch.unsqueeze(disp, -1)

            # This checks if the sizes match up to the final dimension, and the final dimension of disp is of size 1.
            # This is also a homoscedastic case.
            # e.g. rain.size = (10, 2, 3), disp.size = (10, 2, 1)
            elif rain.size()[:-1] == disp.size()[:-1] and disp.size(-1) == 1:  # Heteroscedastic case
                pass

            # If none of the above pass, then the size of disp is incorrect.
            else:
                raise ValueError("disp is of incorrect size")

        # Check validity of reduction mode
        if self.reduction != 'none' and self.reduction != 'mean' and self.reduction != 'sum':
            raise ValueError(self.reduction + " is not valid")

        # Entries of disp must be non-negative
        if torch.any(disp < 0):
            raise ValueError("disp has negative entry/entries")

        # Calculate the hurdle burnouilli loss
        loss_norain = self.bce_logits( logits, did_rain )

        # Calculate the ll-loss
        indices_rainydays = torch.where(did_rain.view( (1,-1))>0)
        
        rain_rainydays = rain.view((1,-1))[indices_rainydays]
        mu_rainydays = mu.view((1,-1))[indices_rainydays]
        disp_rainydays = disp.view((1,-1))[indices_rainydays]

        # continuous loss 
        loss_rain = self.gamma_nll(obs=rain_rainydays, mu=mu_rainydays, disp=disp_rainydays)
        loss_rain = loss_rain.sum()
        loss_rain = self.pos_weight * loss_rain

        if self.reduction == 'mean':
            loss_norain     = loss_norain/rain.numel()
            loss_rain = loss_rain/rain.numel()
            loss = loss_norain + loss_rain 
        
        return loss, {'loss_norain':loss_norain.detach() , 'loss_rain':loss_rain.detach()}            

    def prediction_metrics(self, rain, did_rain, mean, logits, **kwargs):

        pred_rain_bool = torch.where( logits>=0.0, 1.0, 0.0)
        pred_acc = torch.mean(  torch.where( pred_rain_bool==did_rain, 1.0, 0.0) )
        pred_rec = (did_rain*pred_rain_bool).sum() / did_rain.sum() if did_rain.sum()>0 else 1.0

        indices_rainydays = torch.where(did_rain.view( (1,-1))==1.0)

        pred_mse = torch.nn.functional.mse_loss(
            mean.view((1,-1))[indices_rainydays],
            rain.view((1,-1))[indices_rainydays]
            ) #MSE on days it did rain
        

        pred_metrics = {'pred_acc': pred_acc.detach(),
                            'pred_rec': pred_rec.detach(),
                            'pred_mse': pred_mse.detach(),
                             }
        return pred_metrics
        
class CompoundPoissonGammaNLLLoss(_Loss):
    """The CompoundPoisson Distribution.

        The underlying distribution is Gamma distribution

        We use the Tweedie parameterization that uses a mu and variance concept.
        
        In the Normal CompoundPoissonGamma distribution:
            - there are three parameters: \alpha, \beta, \lambda
            - X is distributed Gamma(\alpha, \beta) - \alpha is the shape, \beta is the rate
            - N is distributed Poisson(\lambda)
            
        In the Tweedie parameterisation:
            - there are three parameters \mu, \theta, \p
                :\mu = \lambda \times \frac{\alpha}{\beta} : can be interpreted as the mu of Y (our target variable)
                :\theta = \lambda * \alpha * (1-\alpha) * \beta^(-2) * \mu^-p : can be interpreted as dispersion
                :\p = \alpha \times (1-\alpha)^-1 and 1<p<2 : can be interpreted as the 'distribution parameters'
            - statistics:
                :
            - note: if \mu ==0, then \lambda ==0
                :  \mu is independent of \lambda 
                : \p interpretation
                    : can be interpreted as p = -( q + 1)
                    : as q tends to 1, \lambda tends to 0?? (i.e. Poisson(0) so the minium amount of samples is chosen)
                    : as q tends to 0, then \alpha (shape of gamm distributions) tends to 0, muing mode and mean of tend to 0

    """
    __constants__ = ['full', 'eps', 'reduction']
    full: bool
    eps: float

    def __init__(self, *, full: bool = True, eps: float = 1e-6, reduction: str = 'mean', pos_weight=1, **kwargs ) -> None:
        super(CompoundPoissonGammaNLLLoss, self).__init__(None, None, reduction)
        self.full = full
        self.eps = torch.finfo(torch.float16).eps
        self.bce_logits = torch.nn.BCEWithLogitsLoss(reduction='sum')
        self.register_buffer('pos_weight',torch.tensor([pos_weight]) )
        self.register_buffer('pi',torch.tensor(math.pi) )
        self.register_buffer('e',torch.tensor(math.e) )
        self.tblogger = kwargs.get('tblogger',None)
        
        self.cp_version = kwargs.get('cp_version',2)

        if self.cp_version in [2,3]:
            self.max_j = kwargs.get('max_j', 12)
            self.register_buffer('j', torch.arange(1, self.max_j+1, 1, dtype=torch.float, requires_grad=False).unsqueeze(-1) )

        elif self.cp_version in [4,5]:
            self.j_window_size = kwargs.get('j_window_size', 5)
            self.register_buffer('j_window_size_float', torch.as_tensor( kwargs.get('j_window_size', 5.0), dtype=torch.float) )
            self.register_buffer('j_window', torch.arange(start=-self.j_window_size+1, end=self.j_window_size, step=1, dtype=torch.float, requires_grad=False) )
        
        # Check validity of reduction mode
        if self.reduction not in  ['none','mean','sum']:
            raise ValueError(self.reduction + " is not valid")
        
    def nll_zero(self, mu, disp, p ):
        #L=0
        ll = -mu.pow(2-p) * disp.pow(-1) * (2-p).pow(-1)
        return -ll

    def nll_positive(self, obs, mu, disp, p, **kwargs  ):
        #L>0
        # using approximation from https://www.hindawi.com/journals/jps/2018/1012647/
        
        # mu = mu.clone()
        # with torch.no_grad():
        #     mu.clamp_(min=self.eps)

        lambda_ = l = mu.pow(2-p) / ( disp * (2-p) )
        alpha = a = (2-p) / (p-1)  #from the gamm distribution
        beta = b = disp*(p-1)*mu.pow(p-1)
        L = obs
        theta = disp
        pi = self.pi
        e = self.e
        
        C = L*mu.pow(1-p)*(1-p).pow(-1) - mu.pow(2-p)*(2-p).pow(-1)
        C *=  theta.pow(-1)

        #------------- Version 2 - using 0<j<=48 
        if self.cp_version == 2:
            j = self.j
                       

            A = torch.log(L.pow(-1))
            #  stirling approx of GammaFunc is inaccurate for GammaFunc<1; the approx goes negative while the true value is positive
            # This leads to an error later on when we use log transform.
                    
            #Therefore we used the improved GammaFunc approximation proposed by Gosper
                # This approximates x! better for x between 0 and 1
            Wj = L.pow(j*a) * (p-1).pow(-a*j) * theta.pow(-j*(1+a)) * (2-p).pow(-j) \
                    *((2*j+1/3)*pi).pow(-0.5) * (j/e).pow(-j) \
                    *((2*j*a+1/3)*pi).pow(-0.5) * ((j*a)/e).pow(-j*a) 
        
            # summing from 1 to 48
            B = Wj.sum(dim=0)
            B = torch.log(B)
            
            ll = A + B + C

        #------------- Version 3 - using 0<j<=48 and jensens inequality to convert log(sum(Wj)) to sum(log(Wj))
        # Calculate jmax. Then calculate a range around j an duse this
        elif self.cp_version == 3:
            j=self.j
            A = torch.log(L.pow(-1))

            logW = (j*a)*torch.log(L) + (-a*j)*torch.log(p-1) + (-j*(1+a))*torch.log(theta) + -j*torch.log(2-p) + -torch.lgamma(j+1) + -torch.lgamma(j*a)
            
            #summing from 1 to 48
            B = logW.sum(dim=0)

            ll = A + B + C
            
        #------------- Version 4 - Use a window around J* and stirling approximation (acc use Goson approx)
        elif self.cp_version == 4:

            with torch.no_grad():
                #  jmax is currently calculated using the Stirling formulation. instead use
                #  the onas method to create an estimation for the value of jmax then in 
                #  paper state that this approximation holds significantly beeter for low x
                jmax = L.pow(2-p) * (2-p).pow(-1) * theta.pow(-1)
                jmax = torch.round(jmax)
                jmax = torch.where(jmax<self.j_window_size, self.j_window_size_float , jmax )
                jmax = jmax[:, None].expand( jmax.numel(), (self.j_window_size*2) - 1) #expanding
                j = (jmax + self.j_window).transpose(0,1) # This is a range of j for each index

            A = torch.log(L.pow(-1))

            #Therefore we used the improved GammaFunc approximation proposed by Gosper
            Wj = L.pow(j*a) * (p-1).pow(j*-a) * theta.pow(-j*(1+a)) * (2-p).pow(-j) \
                    *((2*j+1/3)*pi).pow(-0.5) * (j/e).pow(-j) \
                    *((2*j*a+1/3)*pi).pow(-0.5) * ((j*a)/e).pow(-j*a) 
            
            B = Wj.sum(dim=0)
            B = torch.log(B)
            
            ll = A + B + C
            
        #------------- Version 5 - Use a window around J*  and jensens inequality to conver log(sum(Wj)) to sum(log(Wj))
        elif self.cp_version == 5:
            
            A = torch.log(L.pow(-1))

            with torch.no_grad():
                jmax = L.pow(2-p) * (2-p).pow(-1) * theta.pow(-1)
                jmax = torch.round(jmax)
                jmax = torch.where(jmax<self.j_window_size, self.j_window_size_float , jmax )
                jmax = jmax[:, None].expand( jmax.numel(), (self.j_window_size*2) - 1) #expanding
                j = (jmax + self.j_window).transpose(0,1) # This is a range of j for each index

            if False and self.tblogger and self.training:
                _1 = (j*a)*torch.log(L)
                _2 = (j*-a)*torch.log(p-1)
                _3 = (-j*(1+a))*torch.log(theta)
                _4 = -j*torch.log(2-p)
                _5 = -torch.lgamma(j+1)
                _6 = -torch.lgamma(j*a)
                logW = _1 + _2 + _3 + _4 + _5 + _6

                self.tblogger.add_scalars(
                    'B_loss',{
                        '1':_1.detach().mean(),
                        '2':_2.detach().mean(),
                        '3':_3.detach().mean(),
                        '4':_4.detach().mean(),
                        '5':_5.detach().mean(),
                        '6':_6.detach().mean()
                        },
                    global_step = kwargs.get('global_step') )
            else:
                logW = (j*a)*torch.log(L) + (j*-a)*torch.log(p-1) + (-j*(1+a))*torch.log(theta) + -j*torch.log(2-p) +\
                    -torch.lgamma(j+1) + -torch.lgamma(j*a)            #summing over j
            
            B = logW.sum(dim=0)

            ll = A + B + C

        return -ll 

    def forward(self, rain: Tensor, did_rain:Tensor, mu: Tensor, disp: Tensor, p: Tensor, **kwargs) -> Tensor:
        
        # Ensuring correct dtypes 
        if did_rain.dtype != p.dtype:
            did_rain = did_rain.to(p.dtype)

        if rain.dtype != mu.dtype:
            rain = rain.to(mu.dtype)

        # Check disp size
            # If disp.size == rain.size, the case is heteroscedastic and no further checks are needed.
            # Otherwise the case is homoscedastic and we need to expand the last dimension
        if disp.size() != rain.size():

            # If disp is one dimension smaller than rain, but the sizes match otherwise, then this is a homoscedastic case.
            # e.g. rain.size = (10, 2, 3), disp.size = (10, 2)
            # -> unsqueeze disp so that disp.shape = (10, 2, 1)
            # this is done so that broadcasting can happen in the loss calculation
            if rain.size()[:-1] == disp.size():
                disp = torch.unsqueeze(disp, -1)

            # This checks if the sizes match up to the final dimension, and the final dimension of disp is of size 1.
            # This is also a homoscedastic case.
            # e.g. rain.size = (10, 2, 3), disp.size = (10, 2, 1)
            elif rain.size()[:-1] == disp.size()[:-1] and disp.size(-1) == 1:  # Heteroscedastic case
                pass

            # If none of the above pass, then the size of disp is incorrect.
            else:
                raise ValueError("disp is of incorrect size")

        # Entries of disp must be non-negative
        if torch.any(disp < 0):
            raise ValueError("disp has negative entry/entries") 

        # Clamping for stability
        disp = disp.clone()
        # rain = rain.clone()
        p = p.clone()
        with torch.no_grad():
            disp.clamp_(min=self.eps)
            # rain.clamp_(min=self.eps)  
            p.clamp_(min=1+self.eps, max=2-self.eps)

        # Gathering indices to seperate days of no rain from days of rain
        count = did_rain.numel()
        indices_rainydays = torch.where(did_rain.view( (1,-1))>0)
        indices_non_rainydays = torch.where(did_rain.view( (1,-1))==0)

        
        # Gathering seperate tensors for days with and without rain
        rain_rainydays = rain.view((1,-1))[indices_rainydays]
        mu_rainydays = mu.view((1,-1))[indices_rainydays]
        disp_rainydays = disp.view((1,-1))[indices_rainydays]
        p_rainydays = p.view((1,-1))[indices_rainydays]

        rain_non_rainydays = rain.view((1,-1))[indices_non_rainydays]
        mu_non_rainydays = mu.view((1,-1))[indices_non_rainydays]
        disp_non_rainydays = disp.view((1,-1))[indices_non_rainydays]
        p_non_rainydays = p.view((1,-1))[indices_non_rainydays]

        # Calculating loss for non rainy days
        loss_norain = self.nll_zero(mu=mu_non_rainydays, disp=disp_non_rainydays, p = p_non_rainydays  )
        
        # Calculating loss for rainy days
        loss_rain = self.nll_positive(obs= rain_rainydays, mu=mu_rainydays, disp=disp_rainydays, p=p_rainydays, global_step=kwargs.get('global_step') )
        

        loss_rain = self.pos_weight * loss_rain
        loss_rain = loss_rain.sum()
        loss_norain = loss_norain.sum()


        if self.reduction == 'mean':
            loss_norain = loss_norain/count
            loss_rain = loss_rain/count

            loss = loss_rain + loss_norain
                    
        return loss, {'loss_norain':loss_norain.detach() , 'loss_rain':loss_rain.detach()}

    def prediction_metrics(self, rain, did_rain, mean, min_rain_value, **kwargs):
        
        pred_rain_bool = torch.where( mean>min_rain_value, 1.0, 0.0)
        pred_acc = torch.mean(  torch.where( pred_rain_bool==did_rain, 1.0, 0.0) )
        pred_rec = (did_rain*pred_rain_bool).sum() / did_rain.sum()  if did_rain.sum()>0 else 1
       
        
        indices_rainydays = torch.where(did_rain.view( (1,-1))>0)

        # Note: we unscale the value prior to caluclating mse to allow for comparison between models
        pred_mse = torch.nn.functional.mse_loss(
            mean.view((1,-1))[indices_rainydays],
            rain.view((1,-1))[indices_rainydays]
        ) #MSE on days it did rain

        
        pred_metrics = {'pred_acc': pred_acc.detach(),
                            'pred_rec': pred_rec.detach(),
                            'pred_mse': pred_mse.detach(),
                             }
        return pred_metrics