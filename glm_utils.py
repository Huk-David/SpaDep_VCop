import torch
from torch._C import Value
from torch.autograd.grad_mode import set_grad_enabled
from torch.distributions.gamma import Gamma
from torch.distributions.half_normal import HalfNormal
from torch.distributions.log_normal import LogNormal
from torch.distributions.normal import Normal
from torch.distributions.bernoulli import Bernoulli

import torch.functional as F
from torch.nn.modules.loss import GaussianNLLLoss, PoissonNLLLoss
from scipy import stats

from torch.nn.modules.loss import _Loss
from torch import Tensor
from typing import Callable, Optional, Union
# torch.autograd.set_detect_anomaly(True)
import numpy as np

from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.transforms import ExpTransform
from torch.distributions.transformed_distribution import TransformedDistribution
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler, FunctionTransformer, MaxAbsScaler
from loss_utils import *
import regex as re
import distributions

from typing import List, Dict
# from pytorch_lightning.utilities.types import _METRIC
from torch.nn import functional as F

from torch.utils.data._utils.collate import np_str_obj_array_pattern, default_collate_err_msg_format, default_collate
from torch._six import string_classes
import collections

#mu functions
class Inverse(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.eps = 1e-6

    def forward(self, x ):
        
        x = x.clone()
        with torch.no_grad():
            x.clamp_(min=self.eps)

        outp = 1/x

        return outp

class Shift(torch.nn.Module):
    def __init__(self, shift=1) -> None:
        super().__init__()
        # self.shift = shift
        self.register_buffer('shift', torch.tensor([shift]))

    def forward(self, x ):
        x = x + self.shift
        return x

class Multiply(torch.nn.Module):
    def __init__(self, multiple=2) -> None:
        super().__init__()
        self.register_buffer('multiple', torch.tensor([multiple]))

    def forward(self, x ):
        x = x*self.multiple
        return x

class Clamp(torch.nn.Module):
    def __init__(self, lb=1e-6, ub=1-1e-6) -> None:
        super().__init__()
        self.lb = lb
        self.ub = ub

    def forward(self, x ):
        x = x.clamp(self.lb, self.ub)
        return x

class ExponentialActivation(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer('z', torch.tensor([0.0]))
    
    def forward(self, x):
        outp = torch.maximum(self.z.expand(x.shape), torch.exp(x)-1 )
        return outp

class Log(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x):
        outp = torch.log(x)
        return outp

MAP_LINK_INVFUNC = {
    'identity':torch.nn.Identity,
    'relu':torch.nn.ReLU,
    'relu_yshift eps': torch.nn.Sequential( torch.nn.ReLU(), Shift( 1e-6 ) ),
    'relu_xshift1_yshifteps': torch.nn.Sequential( Shift(1), torch.nn.ReLU(), Shift( 1e-6 ) ),
    
    'xshiftn_relu_yshifteps': lambda n: torch.nn.Sequential( Shift(n), torch.nn.ReLU(), Shift( 1e-6 ) ),
    'xshiftn_relu_yshifteps_inverse': lambda n:torch.nn.Sequential( Shift(n), torch.nn.ReLU(), Shift( 1e-4 ), Inverse() ),
    'xshiftn_relu_timesm_yshifteps':lambda shift, mult: torch.nn.Sequential( Shift(shift), torch.nn.ReLU(), Multiply(mult) , Shift( 1e-3 ) ),
    'xshiftn_relu_timesm_yshiftn':lambda xshift, mult, yshift: torch.nn.Sequential( Shift(xshift), torch.nn.ReLU(), Multiply(mult) , Shift( yshift ) ),
    
    'xmult_exponential_yshifteps':lambda mult, mineps : torch.nn.Sequential( Multiply(mult), ExponentialActivation(), Shift(mineps) ),
    'xmult_exponential_yshifteps_log':lambda mult, mineps : torch.nn.Sequential( Multiply(mult), ExponentialActivation(), Shift(mineps), Log() ),

    'relu_inverse':torch.nn.Sequential( torch.nn.ReLU(), Inverse() ),
    'relu_yshifteps_inverse':torch.nn.Sequential( torch.nn.ReLU(), Shift( 1e-4 ) , Inverse() ),
    
    'sigmoid':torch.nn.Sigmoid(),
    'sigmoid_yshiftn': lambda shift: torch.nn.Sequential( torch.nn.Sigmoid(), Shift(shift) ),
    'sigmoid_yshift1': torch.nn.Sequential( torch.nn.Sigmoid(), Shift(1) ),
    'sigmoid_yshift6': torch.nn.Sequential( torch.nn.Sigmoid(), Shift(6) ),
    'sigmoid_clamp_eps': torch.nn.Sequential( torch.nn.Sigmoid(), Clamp( lb=1e-6, ub=1-1e-6 ) ),
    'divn_sigmoid_clampeps_yshiftm': lambda factor, shift : torch.nn.Sequential( Multiply(1/factor), torch.nn.Sigmoid(), Clamp( lb=1e-1, ub=1-1e-1 ), Shift(shift) )     
}
    
# Maps the distribution name to a list of canonical/common inverse link functions. 

MAP_DISTRIBUTION_LINKFUNC = {
    'beta':['sgimoid','sigmoid_clamp_eps'],
    'gamma':['relu','relu_inverse','relu_yshifteps','relu_yshifteps_inverse','sigmoid_clamp_eps'],
    'normal':['relu','relu_inverse','relu_yshifteps'],
    'uniform_positive':['relu_xshift1_yshifteps', 'relu_yshifteps_inverse','relu_yshifteps','relu_xshift0-1_yshifteps','relu_xshift2_times4_yshifteps','relu_xshift0-1_yshifteps_inverse']
}

MAP_NAME_DISTRIBUTION = {
    'normal': Normal ,
    'lognormal': LogNormal ,
    'gamma': Gamma,
    'HalfNormal': HalfNormal,
    'gamma_hurdle':distributions.GammaHurdle,
    'lognormal_hurdle':distributions.LogNormalHurdle,
    'compound_poisson':distributions.CompoundPoisson
}

MAP_DISTRIBUTION_LOSS = {

    'poisson': PoissonNLLLoss, #s(log_input, target, log_input=self.log_input, full=self.full,eps=self.eps, reduction=self.reduction)
    'normal': GaussianNLLLoss, # (input, target, var, full=self.full, eps=self.eps, reduction=self.reduction)
    'lognormal_hurdle':LogNormalHurdleNLLLoss,
    # 'compound_poisson':CPNLLLoss,
    # 'CP_GLM':CPGLMNLLLoss,
    'compound_poisson':CompoundPoissonGammaNLLLoss,
    'gamma_hurdle':GammaHurdleNLLLoss
}

def tuple_type(strings):
    # if isinstance(str, strings):
    if isinstance(strings, tuple):
        return strings

    strings = strings.replace("(", "").replace(")", "")
    if "." in strings:
        mapped_float = map(float, strings.split(","))
        mapped = mapped_float
    else:
        mapped_int = map(int, strings.split(","))
        mapped = mapped_int

    return tuple(mapped)


class GLMMixin:

    def _get_inv_link(self, link_name, params=None):
        
        invfunc =   MAP_LINK_INVFUNC[link_name]
        
        if not hasattr(invfunc, '__dict__') or params:
            invfunc = invfunc(*params)            
        elif not hasattr(invfunc, '__dict__') and params==None:
            raise ValueError(f"params can not be None when using {link_name} activation function")
        return invfunc

    def check_distribution_link(self, distribution_name, link_name):
        bool_check = link_name in MAP_DISTRIBUTION_LINKFUNC.get(distribution_name,[])
        return bool_check

    def _get_distribution(self, distribution_name:str) -> Distribution:
        """Retrieves the distribution class when passed the name of the distributino

        Args:
            distribution_name ([type]): [description]

        Returns:
            [type]: [description]
        """
        return MAP_NAME_DISTRIBUTION[distribution_name]
            
    def _get_dispersion_range(self, distribution_name, **kwargs):

        if distribution_name == "lognormal_hurdle":
            min = kwargs.get('eps',1e-3 )
            max = None
        
        elif distribution_name == "gamma_hurdle":
            #Dispersion should range between 0 and 1, any larger and the gamma distribution places a large weight on 0
            min = None
            max = None
        
        elif distribution_name == "compound_poisson":
            min = kwargs.get('eps',1e-1 )
            max = None
        
        return min, max
    
    def unscale_rain(self, rain_scaled, scaler ):

        if type(scaler) == MaxAbsScaler:
            rain = rain_scaled*scaler.scale_[0]

        elif type(scaler) == MinMaxScaler:
            rain = rain_scaled * torch.as_tensor(1/scaler.scale_, device=rain_scaled.device)
        else:
            raise NotImplementedError
        return rain

    def _get_loglikelihood_loss_func(self,  distribution_name ):
        return MAP_DISTRIBUTION_LOSS[distribution_name]

def _format_checkpoint_name(
    cls,
    filename: Optional[str],
    metrics, #Dict[str, _METRIC],
    prefix: str = "",
    auto_insert_metric_name: bool = True,
    ) -> str:
    if not filename:
        # filename is not set, use default name
        filename = "{epoch}" + cls.CHECKPOINT_JOIN_CHAR + "{step}"

    # check and parse user passed keys in the string
    groups = re.findall(r"(\{.*?)[:\}]", filename)
    if len(groups) >= 0:
        for group in groups:
            name = group[1:]

            if auto_insert_metric_name:
                filename = filename.replace(group, name.replace("/","_") + "={" + name)

            if name not in metrics:
                metrics[name] = 0
        filename = filename.format(**metrics)

    if prefix:
        filename = cls.CHECKPOINT_JOIN_CHAR.join([prefix, filename])

    return filename

def default_collate_concat(batch):
    r"""
        Function that takes in a batch of data and puts the elements within the batch
        into a tensor with an additional outer dimension - batch size. The exact output type can be
        a :class:`torch.Tensor`, a `Sequence` of :class:`torch.Tensor`, a
        Collection of :class:`torch.Tensor`, or left unchanged, depending on the input type.
        This is used as the default function for collation when
        `batch_size` or `batch_sampler` is defined in :class:`~torch.utils.data.DataLoader`.
        Here is the general input type (based on the type of the element within the batch) to output type mapping:
        * :class:`torch.Tensor` -> :class:`torch.Tensor` (with an added outer dimension batch size)
        * NumPy Arrays -> :class:`torch.Tensor`
        * `float` -> :class:`torch.Tensor`
        * `int` -> :class:`torch.Tensor`
        * `str` -> `str` (unchanged)
        * `bytes` -> `bytes` (unchanged)
        * `Mapping[K, V_i]` -> `Mapping[K, default_collate([V_1, V_2, ...])]`
        * `NamedTuple[V1_i, V2_i, ...]` -> `NamedTuple[default_collate([V1_1, V1_2, ...]), default_collate([V2_1, V2_2, ...]), ...]`
        * `Sequence[V1_i, V2_i, ...]` -> `Sequence[default_collate([V1_1, V1_2, ...]), default_collate([V2_1, V2_2, ...]), ...]`
        Args:
            batch: a single batch to be collated
        Examples:
            >>> # Example with a batch of `int`s:
            >>> default_collate([0, 1, 2, 3])
            tensor([0, 1, 2, 3])
            >>> # Example with a batch of `str`s:
            >>> default_collate(['a', 'b', 'c'])
            ['a', 'b', 'c']
            >>> # Example with `Map` inside the batch:
            >>> default_collate([{'A': 0, 'B': 1}, {'A': 100, 'B': 100}])
            {'A': tensor([  0, 100]), 'B': tensor([  1, 100])}
            >>> # Example with `NamedTuple` inside the batch:
            >>> Point = namedtuple('Point', ['x', 'y'])
            >>> default_collate([Point(0, 0), Point(1, 1)])
            Point(x=tensor([0, 1]), y=tensor([0, 1]))
            >>> # Example with `Tuple` inside the batch:
            >>> default_collate([(0, 1), (2, 3)])
            [tensor([0, 2]), tensor([1, 3])]
            >>> # Example with `List` inside the batch:
            >>> default_collate([[0, 1], [2, 3]])
            [tensor([0, 2]), tensor([1, 3])]
    """
    elem = batch[0]
    elem_type = type(elem)

    if isinstance(elem, torch.Tensor):
        out = None
        
        if torch.utils.data.get_worker_info() is not None:
            try:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum(x.numel() for x in batch)
                storage = elem.storage()._new_shared(numel)
            
                out = elem.new(storage).resize_(len(batch), *list(elem.size()[1:]))
                return torch.concat(batch, 0, out=out)
            except Exception as e:
                pass
        res =  torch.concat( batch, 0 )
        return res
    
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))
            return default_collate_concat([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
        # return sum(batch, [ ])
    elif isinstance(elem, collections.abc.Mapping):
        return {key: default_collate_concat([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate_concat(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.

        if isinstance(elem, tuple):
            return [default_collate_concat(samples) for samples in transposed]  # Backwards compatibility.
        else:
            try:
                return elem_type([default_collate_concat(samples) for samples in transposed])
            except TypeError:
                # The sequence type may not support `__init__(iterable)` (e.g., `range`).
                return [default_collate_concat(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))