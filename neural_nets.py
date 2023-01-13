import re
from typing import Union
from numpy import float64
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from sklearn.metrics import jaccard_score
from torch.optim import lr_scheduler
from torch import nn
import torch
from glm_utils import GLMMixin
from transformers.optimization import Adafactor, AdafactorSchedule
#from better_lstm import LSTM
import einops
from einops import rearrange as rearng
from einops import reduce
import pytorch_lightning as pl
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler
import torchtyping
from typing import Tuple
import argparse
from torch.nn import Parameter
import math
#python3 -m pip install git+https://github.com/keitakurita/Better_LSTM_PyTorch.git

class HLSTM(nn.Module):
    
    model_type = "HLSTM"

    def __init__(self,
                    input_shape=(6,),
                    output_shape=(2,),
                    hidden_dim:int=64,
                    num_layers:int=2, 
                    dropoutw:float=0.35,
                    zero_inflated_model:bool=False ) -> None:
        """[summary]

        Args:
            input_shape (tuple, optional): [description]. Defaults to (6,).
            output_shape (tuple, optional): [description]. Defaults to (2,).
            hidden_dim (int, optional): [Dimensions of hidden layers in model]. Defaults to 32.
            num_layers (int, optional): [Number of layers in neural network]. Defaults to 2.
            zero_inflated_model (bool, optional): [Whether or not we use a zero inflated model]. Defaults to False.
        """
        super().__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.zero_inflated_model = zero_inflated_model

        self.upscale = nn.Sequential( 
                            nn.Linear( 
                                *(self.input_shape if len(self.input_shape)==1 else self.input_shape[-1:]), 
                                hidden_dim, bias=False ) )

        self.encoder = nn.Sequential(
                        *[SkipConnectionLSTM(LSTM( input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers//2,
                            dropouti=0.25, dropoutw=dropoutw, dropouto=0.25, batch_first=True, proj_size=hidden_dim//2,
                            bidirectional=True)) for n in range(num_layers//2)],
            ExtractLSTMOutputFeatures(elem="all"))

        self.outp_mu = nn.Sequential(  nn.Linear(hidden_dim, hidden_dim//2, bias=False), 
                                            nn.GELU(),
                                            nn.Linear(hidden_dim//2,
                                                     *(self.output_shape if len(self.output_shape)==1 else self.output_shape[-1:]),
                                                      bias=True))

        self.outp_dispersion = nn.Sequential( nn.Linear(hidden_dim, hidden_dim//2, bias=False), 
                                            nn.GELU(),
                                            nn.Linear(hidden_dim//2,
                                                     *(self.output_shape if len(self.output_shape)==1 else self.output_shape[-1:]),
                                                      bias=True))

        self.outp_logitsrain = nn.Sequential(  nn.Linear(hidden_dim, hidden_dim//2, bias=False),
                                                nn.GELU(),
                                                    nn.Linear(hidden_dim//2,
                                                    *(self.output_shape if len(self.output_shape)==1 else self.output_shape[-1:]),
                                                    bias=True))

    def forward(self, x, standardized_output=True):
        x = self.upscale(x)
        h = self.encoder(x)

        output = {}

        output['mu'] = self.outp_mu(h)
        output['disp'] = self.outp_dispersion(h)

        output['logits'] = self.outp_logitsrain(h)
        
        return output
    
    @staticmethod
    def parse_model_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=True, allow_abbrev=False)

        parser.add_argument("--num_layers", default=4, type=int)
        parser.add_argument("--hidden_dim", default=64, type=int)
        # parser.add_argument("--dropout", default=0.1, type=float)
        parser.add_argument("--dropoutw", default=0.35, type=float)

            
        model_args = parser.parse_known_args()[0]
        return model_args

class HLSTM_tdscale(nn.Module):
    
    model_type = "HLSTM_tdscale"

    def __init__(self,
                    input_shape=(6,),
                    output_shape=(2,),
                    hidden_dim:int=64,
                    num_layers:int=2, 
                    dropout:float=0.1,
                    dropoutw:float=0.35,
                    dropouto:float=0.25,
                    dropouti:float=0.25,

                    zero_inflated_model:bool=False,
                    
                    #Attn params
                    lookback:int = 7,
                    tfactor:int = 4,
                     ) -> None:
        """[summary]

        Args:
            input_shape (tuple, optional): [description]. Defaults to (6,).
            output_shape (tuple, optional): [description]. Defaults to (2,).
            hidden_dim (int, optional): [Dimensions of hidden layers in model]. Defaults to 32.
            num_layers (int, optional): [Number of layers in neural network]. Defaults to 2.
            zero_inflated_model (bool, optional): [Whether or not we use a zero inflated model]. Defaults to False.
        """
        super().__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.zero_inflated_model = zero_inflated_model

        self.upscale = nn.Sequential( 
                            nn.Linear( 
                                *(self.input_shape if len(self.input_shape)==1 else self.input_shape[-1:]), 
                                hidden_dim, bias=False )
                                )

        self.encoder = nn.Sequential(
                        *[SkipConnectionLSTM(LSTM( input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers,
                            dropouti=dropouti, dropoutw=dropoutw, dropouto=dropouto, batch_first=True, proj_size=hidden_dim//2,
                            bidirectional=True)) for n in range( max( num_layers-1, 1 ) )],
            ExtractLSTMOutputFeatures(elem="all")
        )
        
        self.outp_mu_tdscale = TemporalDownScaleAttention( lookback, tfactor, hidden_dim, 4, dropout, False)
        self.outp_mu = nn.Sequential( nn.Linear(hidden_dim, hidden_dim, bias=False), 
                                        nn.GELU(),
                                        nn.Linear(hidden_dim, 
                                                    *(self.output_shape if len(self.output_shape)==1 else self.output_shape[-1:]),
                                                     bias=True)                                        
                                         )

        self.outp_dispersion_tdscale = TemporalDownScaleAttention( lookback,tfactor, hidden_dim, 4, dropout, False)
        self.outp_dispersion = nn.Sequential( nn.Linear(hidden_dim, hidden_dim, bias=False), 
                                            nn.GELU(),
                                            nn.Linear(hidden_dim,
                                                     *(self.output_shape if len(self.output_shape)==1 else self.output_shape[-1:]),
                                                      bias=True) )

        self.p_tdscale = TemporalDownScaleAttention( lookback, tfactor, hidden_dim, 4, dropout, False)

        self.outp_logitsrain = nn.Sequential(  nn.Linear(hidden_dim, hidden_dim, bias=False),
                                                nn.GELU(),
                                                    nn.Linear(hidden_dim,
                                                    *(self.output_shape if len(self.output_shape)==1 else self.output_shape[-1:]),
                                                    bias=True) )

    def forward(self, x, standardized_output=True):
        x = self.upscale(x)
        h = self.encoder(x)

        output = {}

        hm = self.outp_mu_tdscale(h)
        output['mu'] = self.outp_mu(hm)

        hd = self.outp_dispersion_tdscale(h)
        output['disp'] = self.outp_dispersion(hd)

        hp = self.p_tdscale(h)
        output['logits'] = self.outp_logitsrain(hp)
        
        return output
    
    @staticmethod
    def parse_model_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=True, allow_abbrev=False)

        parser.add_argument("--num_layers", default=3, type=int)
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

class TemporalDownScaleAttention(nn.Module):

    def __init__(self,
        num_embeddings,
        tfactor,
        embedding_dim,
        num_heads,
        dropout,
        bias
    ) -> None:
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.tfactor = tfactor
        self.pos_emb = nn.Embedding( self.tfactor,
                            embedding_dim )
        self.register_buffer('indicies', torch.arange(0,self.tfactor, dtype=torch.long))

        self.mha = nn.MultiheadAttention(
                        embedding_dim,
                        num_heads = num_heads,
                        dropout = dropout,
                        bias = bias,
                        batch_first=True,
        )   

    def forward(self, x):
        
        orig_shape = x.shape

        pos_embd = self.pos_emb(self.indicies)

        x = x.reshape(-1, self.tfactor, self.embedding_dim)
        embd = x + pos_embd
        query = embd.mean(dim=-2,keepdim=True)
        outp = self.mha(query=query,
                        key=embd,
                        value=embd,
                        need_weights=False,
                        )[0]
                        
        outp = outp.view( orig_shape[0], self.num_embeddings, self.embedding_dim  )

        return outp


class HConvLSTM_tdscale(nn.Module):
    
    model_type = "HConvLSTM_tdscale"

    def __init__(self,
                input_shape=(6,),
                    output_shape=(2,),
                    hidden_dim:int=64,
                    lstm_cell_kernel_size:Tuple=(1,1),
                    num_layers:int=2, 
                    dropout:float=0.1,

                    dropoutw:float=0.35,
                    dropouto:float=0.25,
                    dropouti:float=0.25,

                    dropout_qk:float=0.1,
                    dropout_v:float=0.1,

                    zero_inflated_model:bool=False,

                    outer_box_dim=[16, 16],
                    
                    heads:int=4,
                    
                    #Attn params
                    lookback:int = 7,
                    tfactor:int = 4,
                     ) -> None:
        """[summary]

        Args:
            input_shape (tuple, optional): [description]. Defaults to (6,).
            output_shape (tuple, optional): [description]. Defaults to (2,).
            hidden_dim (int, optional): [Dimensions of hidden layers in model]. Defaults to 32.
            num_layers (int, optional): [Number of layers in neural network]. Defaults to 2.
            zero_inflated_model (bool, optional): [Whether or not we use a zero inflated model]. Defaults to False.
        """
        super().__init__()

        self.input_shape = input_shape[0] if isinstance(input_shape,Tuple) else input_shape
        self.output_shape = output_shape[0] if isinstance(output_shape,Tuple) else output_shape
        self.outer_box_dim = outer_box_dim
        self.hidden_dim = hidden_dim
        self.zero_inflated_model = zero_inflated_model
        self.tfactor = tfactor

        self.upscale = nn.Conv2d( in_channels=self.input_shape,
                                    out_channels=self.hidden_dim,
                                    kernel_size=(lstm_cell_kernel_size,lstm_cell_kernel_size),
                                    bias=False )

        self.encoder = nn.Sequential(
            *[
                SkipConnectionConvLSTM(
                    ConvLSTM(
                        img_size=outer_box_dim, input_dim=self.hidden_dim,
                        project_dim=hidden_dim//2,
                        hidden_dim=hidden_dim, lstm_cell_kernel_size=(lstm_cell_kernel_size,lstm_cell_kernel_size),
                        cnn_dropout=dropoutw, rnn_dropout=dropoutw,
                        dropouti=dropouti, dropouto=0 if n+1==num_layers else dropouto,
                        batch_first=True,
                        bias=False, peephole=False, layer_norm = False,
                        return_sequence=True, bidirectional=True),
                    
                    # increase_time_steps=2 if n==num_layers-1 else None
                    )
                    for n in range(num_layers)
            ]
        )
               
        self.outp_mu_tdscale = TemporalConvDownScaleAttention( tfactor=tfactor,
                                                                in_channels=self.hidden_dim, 
                                                                in_height=self.outer_box_dim[0], 
                                                                in_width=self.outer_box_dim[1],
                                                                dropout_qk=dropout_qk,
                                                                dropout_v=dropout_v,
                                                                heads=heads)
        self.outp_mu = nn.Sequential( 
                            nn.Conv2d( in_channels=hidden_dim,
                                        out_channels=(self.output_shape+self.input_shape)//2,
                                        kernel_size=3,
                                        padding='same',
                                        bias=True), 
                            nn.GELU(),
                            nn.Conv2d( in_channels=(self.output_shape+self.input_shape)//2,
                                        out_channels=self.output_shape,
                                        kernel_size=1
                                        ,bias=False)                              
                                         )

        self.outp_dispersion_tdscale = TemporalConvDownScaleAttention( tfactor=tfactor,
                                                                in_channels=self.hidden_dim, 
                                                                in_height=self.outer_box_dim[0], 
                                                                in_width=self.outer_box_dim[1],
                                                                dropout_qk=dropout_qk,
                                                                dropout_v=dropout_v,
                                                                heads=heads)
        self.outp_dispersion = nn.Sequential( 
                            nn.Conv2d( in_channels=hidden_dim,
                                        out_channels=(self.output_shape+self.input_shape)//2,
                                        kernel_size=3,
                                        padding='same',
                                        bias=True), 
                            nn.GELU(),
                            nn.Conv2d( in_channels=(self.output_shape+self.input_shape)//2,
                                        out_channels=self.output_shape,
                                        kernel_size=1,
                                        bias=False)                              
                                         )

        self.out_p_tdscale = TemporalConvDownScaleAttention( tfactor=tfactor,
                                                            in_channels=self.hidden_dim, 
                                                            in_height=self.outer_box_dim[0], 
                                                            in_width=self.outer_box_dim[1],
                                                            dropout_qk=dropout_qk,
                                                            dropout_v=dropout_v,
                                                            heads=heads)
        self.outp_logitsrain = nn.Sequential( 
                        nn.Conv2d( in_channels=hidden_dim,
                                    out_channels=(self.output_shape+self.input_shape)//2,
                                    kernel_size=3,
                                    padding='same',
                                    bias=False), 
                        nn.GELU(),
                        nn.Conv2d( in_channels=(self.output_shape+self.input_shape)//2,
                                    out_channels=self.output_shape,
                                    kernel_size=1,
                                    bias=False)          
                                        )
            

    def forward(self, x, standardized_output=True):
        b, t, _, _, _ = x.shape
        x = einops.rearrange(x, 'b t h w c -> (b t) c h w')
        x = self.upscale(x)
        x = einops.rearrange(x, '(b t) c h w -> b t c h w', b=b, t=t)
        
        h, li_finalh_finalc, li_finalh_finalc_reverse = self.encoder(x)
        # h = x

        h = torch.nn.functional.gelu(x)
        
        output = {}

        hm = self.outp_mu_tdscale(h).squeeze(2)
        hm = rearng( hm, 'b t ... -> (b t) ...')
        output['mu'] = rearng( self.outp_mu(hm), '(b t) c1 ... -> b (t c1) ...', b=b, t=t//self.tfactor)

        hd = self.outp_dispersion_tdscale(h).squeeze(2)
        hd = rearng( hd, 'b t ... -> (b t) ...')
        output['disp'] = rearng( self.outp_dispersion(hd), '(b t) c1 ... -> b (t c1) ...', b=b, t=t//self.tfactor)

        hp = self.out_p_tdscale(h).squeeze(2)
        hp = rearng( hp, 'b t ... -> (b t) ...')
        output['logits'] = rearng( self.outp_logitsrain(hp) , '(b t) c1 ... -> b (t c1) ...', b=b, t=t//self.tfactor)
        
        return output
    
    @staticmethod
    def parse_model_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=True, allow_abbrev=False)
        parser.add_argument("--num_layers", default=4, type=int)
        parser.add_argument("--hidden_dim", default=64, type=int)
        parser.add_argument("--dropoutw", default=0.35, type=float)
        parser.add_argument("--dropout_qk", default=0.15, type=float)
        parser.add_argument("--dropout_v", default=0.15, type=float)
        parser.add_argument("--lstm_cell_kernel_size", default=1, type=int)
        parser.add_argument("--heads", default=4, type=float)
        parser.add_argument("--dropouto", default=0.25, type=float)
        parser.add_argument("--dropouti", default=0.25, type=float)
        model_args = parser.parse_known_args()[0]
        return model_args


class SkipConnectionConvLSTM(nn.Module):
    def __init__(self, module:nn.Module, increase_time_steps=None) -> None:
        super().__init__( )

        self.inner_lstm = module
        self.increase_time_steps = increase_time_steps

    def forward(self, inp):
        
        if isinstance(inp, Tuple):
            inp, li_finalh_finalc, li_finalh_finalc_reverse = inp
            outp, li_finalh_finalc, li_finalh_finalc_reverse = self.inner_lstm(inp)
        else:
            outp, li_finalh_finalc, li_finalh_finalc_reverse = self.inner_lstm(inp)

        if self.increase_time_steps:
            outp = outp + inp.repeat_interleave(2, dim=1)
        else: 
            outp = outp + inp     

        return outp, li_finalh_finalc, li_finalh_finalc_reverse


class ConvLSTM(nn.Module):

    """
        '''
        Date: 2020/08/30
        @author: KimUyen
        # The code was revised from repo: https://github.com/ndrplz/ConvLSTM_pytorch
        # The code was revised from repo: https://github.com/KimUyen/ConvLSTM-Pytorch/blob/master/convlstm.py
        '''

            Parameters:
                input_dim: Number of channels in input
                hidden_dim: Number of hidden channels
                kernel_size: Size of kernel in convolutions
                cnn_dropout, rnn_dropout: float
                    cnn_dropout: dropout rate for convolutional input.
                    rnn_dropout: dropout rate for convolutional state.
                batch_first: Whether or not dimension 0 is the batch or not
                bias: Bias or no bias in Convolution
                return_sequence: return output sequence or final output only
                bidirectional: bool
                    bidirectional ConvLSTM
            Input:
                A tensor of size B, T, C, H, W or T, B, C, H, W
            Output:
                A tuple of two sequences output and state
            Example:
                >> x = torch.rand((32, 10, 64, 128, 128))
                >> convlstm = ConvLSTM(input_dim=64, hidden_dim=16, kernel_size=(3, 3), 
                                    cnn_dropout = 0.2,
                                    rnn_dropout=0.2, batch_first=True, bias=False)
                >> output, last_state = convlstm(x)
    """

    def __init__(self, img_size, input_dim, hidden_dim, project_dim ,lstm_cell_kernel_size,
                 cnn_dropout=0.5, rnn_dropout=0.5,  
                 dropouti=0.0, dropouto=0.0,
                 batch_first=True, bias=True, peephole=False,
                 layer_norm=False,
                 return_sequence=True,
                 bidirectional=False):
        super(ConvLSTM, self).__init__()

        self.batch_first = batch_first
        self.return_sequence = return_sequence
        self.bidirectional = bidirectional
        self.dropouti = dropouti
        self.dropouto = dropouto
        self.project_dim =project_dim
        self.hidden_dim = hidden_dim

        cell_fw = ConvLSTMCell(img_size = img_size,
                                 input_dim=input_dim,
                                 hidden_dim=hidden_dim,
                                 kernel_size=lstm_cell_kernel_size,
                                 cnn_dropout=cnn_dropout,
                                 rnn_dropout=rnn_dropout,
                                 bias=bias,
                                 peephole=peephole,
                                 layer_norm=layer_norm)
        self.cell_fw = cell_fw
        self.conv_proj_fw = nn.Conv2d(hidden_dim,self.project_dim,kernel_size=(1,1),
                                        bias=False)
        
        if self.bidirectional is True:
            cell_bw = ConvLSTMCell(img_size = img_size,
                                     input_dim=input_dim,
                                     hidden_dim=hidden_dim,
                                     kernel_size=lstm_cell_kernel_size,
                                     cnn_dropout=cnn_dropout,
                                     rnn_dropout=rnn_dropout,
                                     bias=bias,
                                     peephole=peephole,
                                     layer_norm=layer_norm)
            self.cell_bw = cell_bw
            self.conv_proj_bw = nn.Conv2d(hidden_dim,self.project_dim,
                                        kernel_size=(1,1),
                                        bias=False)
        
        self.dropout_input = nn.Dropout2d( self.dropouti )
        self.dropout_output = nn.Dropout2d( self.dropouto )


    def forward(self, input_tensor, hidden_state=None):
        """
            Parameters
            ----------
            input_tensor: todo
                5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
            hidden_state: todo
                None. todo implement stateful
            Returns
            -------
            layer_output, last_state
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, seq_len, _, _h, _w = input_tensor.size()

        input_tensor  = self.dropout_input(input_tensor.reshape(b*seq_len, _, _h, _w)).reshape(b, seq_len, _, _h, _w)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state, hidden_state_inv = self._init_hidden(batch_size=b)
            # if self.bidirectional is True:
            #     hidden_state_inv = self._init_hidden(batch_size=b)

        ## LSTM forward direction
        input_fw = input_tensor
        h, c = hidden_state
        output_inner = []
        
        #TODO: need to optimized this because right now this for loop slow
        for t in range(seq_len):
            h, c = self.cell_fw(input_tensor=input_fw[:, t, :, :, :],
                                             cur_state=[h, c])
            
            output_inner.append(h)
        output_inner = torch.stack((output_inner), dim=1)
        
        layer_output = output_inner
        layer_output = self.conv_proj_fw( layer_output.reshape(b*seq_len, _, _h, _w ) ) 
        layer_output = einops.rearrange( layer_output, '(b s) c h w -> b s c h w', b=b, s=seq_len)
        # layer_output  = self.dropout_output(layer_output)

        h = layer_output[:, -1, :, :, :]
       
        last_state = [h, c]
        
        ####################
        
        
        ## LSTM inverse direction
        if self.bidirectional is True:
            input_inv = input_tensor
            h_inv, c_inv = hidden_state_inv
            output_inv = []
            for t in range(seq_len-1, -1, -1):
                h_inv, c_inv = self.cell_bw(input_tensor=input_inv[:, t, :, :, :],
                                                 cur_state=[h_inv, c_inv])
                output_inv.append(h_inv)
            output_inv.reverse() 
            output_inv = torch.stack((output_inv), dim=1)


            # output_inv = self.conv_proj_bw(output_inv)
            output_inv = self.conv_proj_bw( output_inv.reshape(b*seq_len, _, _h, _w ) )
            output_inv  = self.dropout_output(output_inv)
            output_inv = einops.rearrange( output_inv, '(b s) c h w -> b s c h w', b=b, s=seq_len)          
            
            
            h_inv = output_inv[:, 0, :, :, :]
            

            layer_output = torch.cat((layer_output, output_inv), dim=2)
            # layer_output  = self.dropout_output(layer_output)
            
            last_state_inv = [h_inv, c_inv]
        ###################################
        
        # return layer_output if self.return_sequence is True else layer_output[:, -1:], last_state, last_state_inv if self.bidirectional is True else None

        layer_output = layer_output if self.return_sequence is True else layer_output[:, -1:] #All outputs or just final layer outputs
        last_state_inv = last_state_inv if self.bidirectional is True else None

        return layer_output, last_state, last_state_inv

    def _init_hidden(self, batch_size):
        init_states_fw = self.cell_fw.init_hidden(batch_size)
        init_states_bw = None
        if self.bidirectional is True:
            init_states_bw = self.cell_bw.init_hidden(batch_size)
        return init_states_fw, init_states_bw


class ConvLSTMCell(nn.Module):

    def __init__(self, img_size, input_dim, hidden_dim, kernel_size, 
                 cnn_dropout, rnn_dropout, bias=True, peephole=False,
                 layer_norm=False):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel for both cnn and rnn.
        cnn_dropout, rnn_dropout: float
            cnn_dropout: dropout rate for convolutional input.
            rnn_dropout: dropout rate for convolutional state.
        bias: bool
            Whether or not to add the bias.
        peephole: bool
            add connection between cell state to gates
        layer_norm: bool
            layer normalization 
        """

        super(ConvLSTMCell, self).__init__()
        self.input_shape = img_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = (int(self.kernel_size[0]/2), int(self.kernel_size[1]/2))
        self.stride = (1, 1)
        self.bias = bias
        self.peephole = peephole
        self.layer_norm = layer_norm
        
        self.out_height = int((self.input_shape[0] - self.kernel_size[0] + 2*self.padding[0])/self.stride[0] + 1)
        self.out_width = int((self.input_shape[1] - self.kernel_size[1] + 2*self.padding[1])/self.stride[1] + 1)
        
        self.input_conv = nn.Conv2d(in_channels=self.input_dim, out_channels=4*self.hidden_dim,
                                  kernel_size=self.kernel_size,
                                  stride = self.stride,
                                  padding=self.padding,
                                  bias=self.bias)
        self.rnn_conv = nn.Conv2d(self.hidden_dim, out_channels=4*self.hidden_dim, 
                                  kernel_size = self.kernel_size,
                                  padding=(math.floor(self.kernel_size[0]/2), 
                                         math.floor(self.kernel_size[1]/2)),
                                  bias=self.bias)
        
        if self.peephole is True:
            self.weight_ci = HadamardProduct((1, self.hidden_dim, self.out_height, self.out_width))
            self.weight_cf = HadamardProduct((1, self.hidden_dim, self.out_height, self.out_width))
            self.weight_co = HadamardProduct((1, self.hidden_dim, self.out_height, self.out_width))
            self.layer_norm_ci = nn.LayerNorm([self.hidden_dim, self.out_height, self.out_width])
            self.layer_norm_cf = nn.LayerNorm([self.hidden_dim, self.out_height, self.out_width])
            self.layer_norm_co = nn.LayerNorm([self.hidden_dim, self.out_height, self.out_width])
        
            
        self.cnn_dropout = nn.Dropout(cnn_dropout)
        self.rnn_dropout = nn.Dropout(rnn_dropout)
        
        self.layer_norm_x = nn.LayerNorm([4*self.hidden_dim, self.out_height, self.out_width])
        self.layer_norm_h = nn.LayerNorm([4*self.hidden_dim, self.out_height, self.out_width])
        self.layer_norm_cnext = nn.LayerNorm([self.hidden_dim, self.out_height, self.out_width])
    
    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        x = self.cnn_dropout(input_tensor)
        x_conv = self.input_conv(x)
        if self.layer_norm is True:
            x_conv = self.layer_norm_x(x_conv)
        # separate i, f, c o
        x_i, x_f, x_c, x_o = torch.split(x_conv, self.hidden_dim, dim=1)
        
        h = self.rnn_dropout(h_cur)
        h_conv = self.rnn_conv(h)
        if self.layer_norm is True:
            h_conv = self.layer_norm_h(h_conv)
        # separate i, f, c o
        h_i, h_f, h_c, h_o = torch.split(h_conv, self.hidden_dim, dim=1)
        

        if self.peephole is True:
            f = torch.sigmoid((x_f + h_f) +  self.layer_norm_cf(self.weight_cf(c_cur)) if self.layer_norm is True else self.weight_cf(c_cur))
            i = torch.sigmoid((x_i + h_i) +  self.layer_norm_ci(self.weight_ci(c_cur)) if self.layer_norm is True else self.weight_ci(c_cur))
        else:
            f = torch.sigmoid((x_f + h_f))
            i = torch.sigmoid((x_i + h_i))
        
        
        g = torch.tanh((x_c + h_c))
        c_next = f * c_cur + i * g
        if self.peephole is True:
            o = torch.sigmoid(x_o + h_o + self.layer_norm_co(self.weight_co(c_cur)) if self.layer_norm is True else self.weight_co(c_cur))
        else:
            o = torch.sigmoid((x_o + h_o))
        
        if self.layer_norm is True:
            c_next = self.layer_norm_cnext(c_next)
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size):
        height, width = self.out_height, self.out_width
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.input_conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.input_conv.weight.device))


class HadamardProduct(nn.Module):
    def __init__(self, shape):
        super(HadamardProduct, self).__init__()
        self.weights = nn.Parameter(torch.rand(shape))
        
    def forward(self, x):
        return x*self.weights

class TemporalConvDownScaleAttention(nn.Module):

    def __init__(self,
        # num_embeddings,
        tfactor,
        in_channels, in_height, in_width,
        dropout_qk = 0.05,
        dropout_v = 0.1,
        heads = 8   

        ) -> None:
        super().__init__()

        # self.num_embeddings = num_embeddings
        self.tfactor = tfactor
        h,w,c = in_height, in_width, in_channels
        self.embedding_dim = h*w
        self.pos_emb = nn.Embedding( self.tfactor, self.embedding_dim) 


        self.register_buffer('indicies', torch.arange(0,self.tfactor, dtype=torch.long))

        self.mha = MultiHeadConvAttn(
                        in_dim=in_channels,
                        q_k_dim = heads,
                        output_dim=in_channels,
                        tfactor = self.tfactor,
                        dropout_qk = dropout_qk,
                        dropout_v = dropout_v
                        )   

    def forward(self, x):
        
        b, t, c, h, w = x.shape

        pos_embd = self.pos_emb(self.indicies)
        pos_embd = einops.rearrange(pos_embd,'i (h w) -> i 1 h w', h=h, w=w)
        

        # Collapsing batch and timestep dimensions
        x = einops.rearrange(x, 'b (t1 i) c h w -> (b t1) i c h w', t1=t//self.tfactor, i=self.tfactor)
        embd = x + pos_embd

        query = embd.mean(dim=-4, keepdim=True) # Averaging in the time dimension

        outp = self.mha(query=query,
                        key=embd,
                        value=embd,
                        need_weights=False)[0]

        outp = einops.rearrange(outp, '(b t1) ol ... -> b (t1 ol) ...', b=b, t1=t//self.tfactor)
        
        
        
        outp = einops.rearrange(query, '(b t1) ol ... -> b (t1 ol) ...', b=b, t1=t//self.tfactor)

        # DEBUG
        # x = einops.rearrange(x, 'b (t1 i) c h w -> b t1 i c h w', t1=t//self.tfactor, i=self.tfactor)
        # outp = x.mean(dim=-4 )

        return outp

class MultiHeadConvAttn(nn.Module):

    """ Self attention Layer

    """
    def __init__(self,
                    in_dim,
                    q_k_dim, #This effectively becomes the multiheads
                    output_dim,
                    tfactor,
                    dropout_qk=0.05,
                    dropout_v = 0.1):
        

        super(MultiHeadConvAttn,self).__init__()

        self.chanel_in = in_dim
        self.tfactor = tfactor
        self.q_k_dim = q_k_dim
        
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=q_k_dim, kernel_size=2, stride=2, padding='valid', bias=False)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = q_k_dim, kernel_size= 2, stride=2, padding='valid', bias=False)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = output_dim , kernel_size= 1)
        self.dropout_qk = nn.Dropout2d(dropout_qk)
        self.dropout_v = nn.Dropout2d(dropout_v)
        self.gamma = nn.Parameter(torch.full((1,),0.5), requires_grad=True) #TODO: change starting value away from zero

        self.linear = nn.Linear(self.q_k_dim, 1, bias=False)

        self.softmax  = nn.Softmax(dim=-1) #

    def forward(self,
                query,
                key,
                value,
                need_weights=False):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)

            
        """
        b, t, c, h, w = value.size()
        _, out_len, _, _, _ = query.shape

        query = einops.rearrange(query, 'b t ... -> (b t) ...')
        key = einops.rearrange(key, 'b1 t1 ... -> (b1 t1) ...')
        value = einops.rearrange(value, 'b t ... -> (b t) ...')
        
        #calculating query and key
        proj_query  = self.query_conv(query) # b, c, h2, w2
        proj_key =  self.key_conv(key) #  b*t, c, h2, w2
        proj_value = self.value_conv(value) # b*t, c1, h1, w1

        # 2D dropout
        proj_query = self.dropout_qk(proj_query)
        proj_key = self.dropout_qk(proj_key)
        proj_value = self.dropout_v(proj_value)

        # collapsing h,w in q and k
        proj_query = einops.rearrange(proj_query,'b c2 h2 w2 -> (b c2) 1 (h2 w2)')
        proj_key = einops.rearrange(proj_key,'(b t) c2 h2 w2 -> (b c2) t (h2 w2)', t=4)
        
        _, c1, h1, w1 = proj_value.shape # b*t, c1, h1, w1
        proj_value = rearng(proj_value, '(b t) c1 h1 w1 -> b t (c1 h1 w1)', t=self.tfactor)

        #Dividing query and key into the seperate heads
        # proj_query = einops.rearrange(proj_query, 'bt c h w -> bt nh hd (h w)', nh=self.num_heads, d=self.head_dim )

        #calculating logits
        energy = torch.bmm(proj_query, proj_key.transpose(-2,-1)  ) # b*c2 , 1 , t
        energy = energy / math.sqrt(self.q_k_dim)

        #softmaxing to get atttention
        attention = torch.nn.functional.softmax(energy, dim=-1) # b*c2, 1 , t

        #reducing heads down to single head via linear layer        
        attention = rearng(attention, '(b c2) ol t -> b (ol c2) t', c2=self.q_k_dim, ol=out_len, t =self.tfactor) #b, ol*c2, t
        
        out = torch.bmm(attention, proj_value ) # b, ol*c2, c1*h1*w1
        
        out = self.linear( rearng(out,'b (ol c2) d -> b (ol d) c2', c2=self.q_k_dim, ol=out_len) ) #b (out_len, d) 1

        out = rearng(out, 'b (ol c1 h1 w1) 1 -> b ol c1 h1 w1', ol=out_len, c1=c1, h1=h1 , w1=w1)
                
        out = self.gamma*out + (1-self.gamma)*reduce( value, '(b t) c1 h1 w1 -> b 1 c1 h1 w1', 'mean', t=t )
        
        #putting attn in corret output shape
        attention = rearng(attention, 'b (ol c2) t -> b c2 ol t', ol=out_len, c2=self.q_k_dim, t=self.tfactor)
        return out, attention


MAP_NAME_NEURALMODEL = {
    'HLSTM': HLSTM,
    'HLSTM_tdscale':HLSTM_tdscale,
    'HConvLSTM_tdscale':HConvLSTM_tdscale,
}
