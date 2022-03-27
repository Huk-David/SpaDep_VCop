from typing import Union
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torch.optim import lr_scheduler
from torch import nn
from glm_utils import GLMMixin
from transformers.optimization import Adafactor, AdafactorSchedule
from better_lstm import LSTM
import einops
import pytorch_lightning as pl
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler
import torchtyping
from typing import Tuple
import argparse
from torch.nn import Parameter
#python3 -m pip install git+https://github.com/keitakurita/Better_LSTM_PyTorch.git

class HLSTM(nn.Module):
    
    model_type = "HLSTM"

    def __init__(self,
                    input_shape=(6,),
                    output_shape=(2,),
                    hidden_dim:int=64,
                    num_layers:int=2, 
                    dropoutw:float=0.35,
                    p_variable_model:bool=False,
                    zero_inflated_model:bool=False ) -> None:
        """[summary]

        Args:
            input_shape (tuple, optional): [description]. Defaults to (6,).
            output_shape (tuple, optional): [description]. Defaults to (2,).
            hidden_dim (int, optional): [Dimensions of hidden layers in model]. Defaults to 32.
            num_layers (int, optional): [Number of layers in neural network]. Defaults to 2.
            p_variable_model (bool, optional): [Whether or not we use a p variable type model]. Defaults to False.
            zero_inflated_model (bool, optional): [Whether or not we use a zero inflated model]. Defaults to False.
        """
        super().__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.p_variable_model = p_variable_model
        self.zero_inflated_model = zero_inflated_model

        self.upscale = nn.Sequential( nn.Linear( input_shape[0], hidden_dim, bias=False ) )


        self.encoder = nn.Sequential(
                        *[SkipConnectionLSTM(LSTM( input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers//2,
                            dropouti=0.25, dropoutw=dropoutw, dropouto=0.25, batch_first=True, proj_size=hidden_dim//2,
                            bidirectional=True)) for n in range(num_layers//2)],
            ExtractLSTMOutputFeatures(elem="all")
        )

        self.outp_mu = nn.Sequential( nn.Linear(hidden_dim, hidden_dim, bias=False), nn.GELU(), nn.Linear(hidden_dim, *self.output_shape, bias=True) )
        self.outp_dispersion = nn.Sequential( nn.Linear(hidden_dim, hidden_dim, bias=False), nn.GELU(), nn.Linear(hidden_dim, *self.output_shape, bias=True) )

        if self.p_variable_model:
            self.outp_logitsrain = nn.Sequential(  nn.Linear(hidden_dim, hidden_dim, bias=False), nn.GELU(), nn.Linear(hidden_dim, *self.output_shape, bias=True) )

    def forward(self, x, standardized_output=True):
        x = self.upscale(x)
        h = self.encoder(x)

        output = {}

        output['mu'] = self.outp_mu(h)
        output['disp'] = self.outp_dispersion(h)

        if self.p_variable_model:
            output['logits'] = self.outp_logitsrain(h)
        
        return output
    
    @staticmethod
    def parse_model_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=True, allow_abbrev=False)

        parser.add_argument("--num_layers", default=4, type=int)
        parser.add_argument("--hidden_dim", default=64, type=int)
        parser.add_argument("--dropoutw", default=0.35, type=float)

            
        model_args = parser.parse_known_args()[0]
        return model_args

class ExtractLSTMOutputFeatures(nn.Module):
    """
        Module that extracts the hidden state output from an LSTM based layer in torch
    """
    def __init__(self, bidirectional=True, elem='final') -> None:
        super().__init__() 

        self.bidirectional = bidirectional
        self.elem = elem

    def forward(self,x):
        out , _ = x

        # # If Bi-Directional LSTM concatenating the Directions dimension
        # if self.bidirectional:
        #     out = einops.rearrange( out, 'b t d h -> b t (d h)')

        if self.elem == 'all':
            pass
        elif self.elem == 'final':
            out = out[:, -1:, :]

        return out

class SkipConnectionLSTM(nn.Module):
    def __init__(self, module:nn.Module) -> None:
        super().__init__()

        self.inner_lstm = module

    def forward(self, inp):
        
        if isinstance(inp, Tuple):
            inp, inp_h = inp
            outp, hx = self.inner_lstm(inp)
        else:
            outp, hx = self.inner_lstm(inp)

        if outp.shape[-1] > inp.shape[-1] and self.inner_lstm.bidirectional:
            outp = outp + inp.repeat(1, 1 ,2)
        else: 
            outp = outp + inp     

        return outp, hx


MAP_NAME_NEURALMODEL = {
    'HLSTM': HLSTM
}
