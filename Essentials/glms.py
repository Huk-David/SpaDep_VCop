from typing import Union
import torch
from torch._C import Value
from torch.autocast_mode import autocast
# from torch.optim import lr_scheduler
from torch import nn
from glm_utils import GLMMixin
from transformers.optimization import Adafactor, AdafactorSchedule
import pickle
import os
from better_lstm import LSTM
import einops
import pytorch_lightning as pl
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler
import torchtyping
import argparse
from neural_nets import MAP_NAME_NEURALMODEL
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from pathlib import Path
import yaml
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from typing import Dict, Any
import json
#python3 -m pip install git+https://github.com/keitakurita/Better_LSTM_PyTorch.git

def tuple_type(strings):
    strings = strings.replace("(", "").replace(")", "")
    if "." in strings:
        mapped_float = map(float, strings.split(","))
        mapped = mapped_float
    else:
        mapped_int = map(int, strings.split(","))
        mapped = mapped_int

    return tuple(mapped)

class NeuralDGLM(pl.LightningModule, GLMMixin):
    
    glm_type = "DGLM"

    def __init__(self,

            scaler_features:Union[MinMaxScaler,StandardScaler],
            scaler_target:Union[MinMaxScaler,StandardScaler],

            nn_name:str=None,
            nn_params:Dict={},
            target_distribution_name = 'lognormal_hurdle',
            
            mu_distribution_name='normal',
            mu_link_name='identity',
            mu_params=None,
            
            dispersion_distribution_name='gamma',
            dispersion_link_name='negative_inverse',
            disp_params=None,

            p_link_name='sigmoid',
            p_params = None, 

            pos_weight=1,
            save_hparams=True,
            debugging = False,

            min_rain_value= 1.0,
            **kwargs):

        super().__init__()

        # Load Neural Model
        neural_net_class = MAP_NAME_NEURALMODEL[nn_name]
        neural_net = neural_net_class( **nn_params )
        self.neural_net = neural_net

        # Restraints on predictions for MaxMin standardized target
        self.scaler_target = scaler_target
        self.scaler_features = scaler_features
        
        # Saving specific hyper-parameters to hparams file
        # this should only run if you are about to train model, atm it runs whenever you load model
        ignore_list = ['save_hparams', "scaler_features","scaler_target","debugging"]
        if save_hparams:
            self.save_hyperparameters(ignore=ignore_list)

        # Target Distribution
        self.target_distribution_name = target_distribution_name
        self.target_distribution = self._get_distribution( self.target_distribution_name )() #target distribution
        self.loss_fct = self._get_loglikelihood_loss_func( target_distribution_name )( pos_weight=pos_weight, **kwargs )  #loss function

        # Checking compatibility of target distribution and link functions
        # assert self.check_distribution_link(mu_distribution_name, mu_link_name), "Incompatible mu link function chosen for target distribution"
        # assert self.check_distribution_link(dispersion_distribution_name, dispersion_link_name),  "Incompatible dispersion link function chosen for target distribution"

        # mu term distribution/link name / inverse_link function
        self.mu_distribution_name = mu_distribution_name
        self.mu_link_name = mu_link_name
        self.mu_inverse_link_function = self._get_inv_link(self.mu_link_name, mu_params)
        
        # Dispersion term distribution/link name / inverse_link function
        self.dispersion_distribution_name = dispersion_distribution_name
        self.dispersion_link_name =  dispersion_link_name
        self.dispersion_inverse_link_function  = self._get_inv_link(self.dispersion_link_name, disp_params)

        self.scaler_features = scaler_features
        self.scaler_target = scaler_target
        self.register_buffer('target_scale', torch.as_tensor(scaler_target.scale_[0]))
        self.register_buffer('min_rain_value', torch.as_tensor(min_rain_value))

        # This mu and max needs to be dependent on the scaler
        self.min_dispersion_output_standardized, self.max_disperion_output_standardized = self._get_dispersion_range( distribution_name=self.target_distribution_name )

        # Setting on hurdle neural_net 
        if self.neural_net.p_variable_model:
            self.p_link_name = p_link_name
            self.p_inverse_link_function = self._get_inv_link(self.p_link_name, p_params)
        
        # reference to logger in case of debugging
        self.debugging = debugging
        
    def forward(self, x ):
        output = self.neural_net(x)

        mu = self.mu_inverse_link_function(output['mu'])
        disp = self.dispersion_inverse_link_function(output['disp'])
        if self.neural_net.p_variable_model:
            output['p'] = self.p_inverse_link_function(output['logits']) 
          
        disp = disp.clone()
        with torch.no_grad():
            if self.min_dispersion_output_standardized or self.max_disperion_output_standardized:
                disp.clamp_(self.min_dispersion_output_standardized, self.max_disperion_output_standardized)

        mu = mu.squeeze(-1)
        disp = disp.squeeze(-1)
    
        output['mu'] = mu
        output['disp'] = disp
        
        return output

    def step(self, batch, step_name ):
        
        inp, target = batch
        target_did_rain, target_rain_value = torch.unbind(target,-1)

        output  = self.forward(inp)
        pred_mu = output['mu']
        pred_disp = output['disp'] 

         
        if self.neural_net.p_variable_model:
            pred_p = output['p'].squeeze(-1)  
            logits = output['logits'].squeeze(-1)     
            loss, composite_losses = self.loss_fct( target_rain_value, target_did_rain, pred_mu, pred_disp, logits=logits, p=pred_p, global_step=self.global_step )

            pred_mu_unscaled, pred_disp_unscaled, _ = self.target_distribution.unscale_distribution( pred_mu.detach(), pred_disp.detach(),
                                                        pred_p.detach(), self.scaler_target )

            pred_mean = self.target_distribution.get_mean( pred_mu_unscaled, pred_disp_unscaled, pred_p )

            rain_unscaled = self.unscale_rain(target_rain_value, self.scaler_target)

            pred_metrics = self.loss_fct.prediction_metrics(rain_unscaled, target_did_rain, pred_mean,
                                                             logits=logits, p=pred_p, min_rain_value=self.min_rain_value )

        else:
            loss = self.loss_fct( target_rain_value, pred_mu, pred_disp )
            composite_losses = None
            pred_metrics = None

        if self.debugging and step_name=='train':
            tblogger = self.trainer.logger.experiment
            global_step = self.trainer.global_step
            tblogger.add_histogram('mu', pred_mu, global_step=global_step)
            tblogger.add_histogram('disp', pred_disp, global_step=global_step)
            if self.neural_net.p_variable_model: 
                tblogger.add_histogram('p', output['p'], global_step=global_step )

        if step_name in ['train','val']:
            return {'loss':loss, 'composite_losses':composite_losses, 'pred_metrics':pred_metrics }

        elif step_name in ['test']:
            output =  {'loss':loss, 'pred_mu':pred_mu, 'pred_disp':pred_disp,  
                            'target_did_rain':target_did_rain, 'target_rain_value':target_rain_value, 'pred_metrics':pred_metrics }

            if self.neural_net.p_variable_model:
                output['composite_losses'] = composite_losses
                output['pred_p'] = pred_p
            return output

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        output = self.step(batch, "train")
        self.log("train_loss/loss",output['loss'])

        if output.get( 'composite_losses', None):
            self.log("train_loss/norain",output['composite_losses']['loss_norain'])
            self.log("train_loss/rain",output['composite_losses']['loss_rain'])
        
        if output.get( 'pred_metrics', None):
            self.log("train_metric/acc",output['pred_metrics']['pred_acc'],  on_step=False, on_epoch=True )
            self.log("train_metric/recall",output['pred_metrics']['pred_rec'], on_step=False, on_epoch=True )
            self.log("train_metric/mse_rain",output['pred_metrics']['pred_mse'],  on_step=False, on_epoch=True )

        return output
    
    def validation_step(self, batch, batch_idx):
        output  = self.step(batch, "val")
        self.log("val_loss/loss", output['loss'], prog_bar=True, on_epoch=True, on_step=False)

        if output.get('composite_losses', None):
            self.log("val_loss/norain", output['composite_losses']['loss_norain'], on_step=False, on_epoch=True)
            self.log("val_loss/rain", output['composite_losses']['loss_rain'], on_step=False, on_epoch=True)

        if output.get( 'pred_metrics', None):
            self.log("val_metric/acc",output['pred_metrics']['pred_acc'] , on_step=False, on_epoch=True)
            self.log("val_metric/recall",output['pred_metrics']['pred_rec'] , on_step=False, on_epoch=True)
            self.log("val_metric/mse_rain",output['pred_metrics']['pred_mse'] , on_step=False, on_epoch=True)

        return output

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:

        res = super().on_save_checkpoint(checkpoint)
        next( filter( lambda cb: isinstance(cb, ModelCheckpoint), self.trainer.callbacks) ).to_yaml()
        return res

    def test_step(self, batch, batch_idx):

        output = self.step(batch, "test")

        # Logging the aggregate loss during testing
        self.log("test_loss/loss", output['loss'])

        if output.get('composite_losses', None):
            self.log("test_loss/norain", output['composite_losses']['loss_norain'], on_epoch=True, prog_bar=True)
            self.log("test_loss/rain", output['composite_losses']['loss_rain'], on_epoch=True, prog_bar=True)

        if output.get( 'pred_metrics', None):
            self.log("test_metric/acc",output['pred_metrics']['pred_acc'] , on_step=False, on_epoch=True)
            self.log("test_metric/recall",output['pred_metrics']['pred_rec'] , on_step=False, on_epoch=True)
            self.log("test_metric/mse_rain",output['pred_metrics']['pred_mse'] , on_step=False, on_epoch=True, )
        
        return output

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        
        # Saving the predictions and targets from the test sets
        pred_mu = torch.cat( [output['pred_mu'] for output in outputs], dim=0 )
        pred_disp = torch.cat( [output['pred_disp'] for output in outputs], dim=0 )

        if self.neural_net.p_variable_model:
            pred_p = torch.cat( [output['pred_p'] for output in outputs], dim=0 )
        else:
            pred_p = None
        
        #target data
        target_did_rain = torch.cat( [output['target_did_rain'] for output in outputs], dim=0 )
        target_rain_value = torch.cat( [output['target_rain_value'] for output in outputs], dim=0 )
        
        #unscaling predictive distribution
        pred_mu_unscaled, pred_disp_unscaled, pred_p_unscaled = self.target_distribution.unscale_distribution( pred_mu, pred_disp, pred_p, self.scaler_target )

        pred_mu_unscaled = pred_mu_unscaled.cpu().numpy()
        pred_disp_unscaled = pred_disp_unscaled.cpu().numpy()
        pred_p_unscaled=pred_p_unscaled.cpu().numpy()

        target_rain_unscaled = self.unscale_rain(target_rain_value, self.scaler_target)
        target_rain_unscaled = target_rain_unscaled.cpu().numpy()

        # Split predictions by location
        test_dl = self.trainer.test_dataloaders[0]

        locations = [ ds.location for ds in test_dl.dataset.datasets]
        cumulative_sizes = [0] + test_dl.dataset.cumulative_sizes
        
        dict_location_data = {}
        for idx, loc in enumerate(locations):
            if cumulative_sizes==0:
                continue
            s_idx = cumulative_sizes[idx]
            e_idx = cumulative_sizes[idx+1]

            lookback = test_dl.dataset.datasets[idx].lookback
            dates = test_dl.dataset.datasets[idx].dates #dates in dataset that are valid
            indexes_filtrd = test_dl.dataset.datasets[idx].indexes_filtrd #indexes_filtrd
            
            date_windows = [ dates[idx-lookback:idx] for idx in indexes_filtrd[lookback: ] ]

            data = {'pred_mu':pred_mu_unscaled[s_idx:e_idx],
                    'pred_disp':pred_disp_unscaled[s_idx:e_idx],
                    'target_did_rain':target_did_rain[s_idx:e_idx],
                    'target_rain_value':target_rain_unscaled[s_idx:e_idx],
                    'date':date_windows }
            
        
            if self.neural_net.p_variable_model:
                data['pred_p'] = pred_p_unscaled[s_idx:e_idx]

            dict_location_data[loc] = data

        dir_path = os.path.dirname( next( ( callback for callback in self.trainer.callbacks if type(callback)==pl.callbacks.model_checkpoint.ModelCheckpoint) ).dirpath )
        file_path = os.path.join( dir_path, "test_output.pkl" ) 
        with open( file_path, "wb") as f:
            pickle.dump( dict_location_data, f )
        
        return super().test_epoch_end(outputs)
        
    def configure_optimizers(self):
        optimizer = Adafactor(self.parameters(), scale_parameter=True, relative_step=True, 
                                warmup_init=True, lr=None, 
                                weight_decay=0.1,
                                clip_threshold=0.5
                                 )
        lr_scheduler = AdafactorSchedule(optimizer)
        return { 'optimizer':optimizer, 
                    'lr_scheduler':lr_scheduler }

    def save_scalers(self, dir_path:str ) -> None:
        os.makedirs(dir_path, exist_ok=True)
        if hasattr(self, 'scaler_features'):
            pickle.dump(self.scaler_features,open(os.path.join(dir_path,"scaler_features.pkl"),"wb") )

        if hasattr(self, 'scaler_target'):
            pickle.dump(self.scaler_features,open(os.path.join(dir_path,"scaler_target.pkl"),"wb") )
    
    @classmethod
    def load_scalers(cls, dir_path:str=None):
        """Loads the scaler object from a given path.
                The path can be a path to the directory containing both the feature and target scaler
                Alternatively, it can be two paths to the each of the feature and target.

        Args:
            dir_path (str, optional): [description]. Defaults to None.


        Returns:
            [type]: [description]
        """

        path_scaler_features = os.path.join(dir_path, "scaler_features.pkl")
        path_scaler_target = os.path.join(dir_path, "scaler_target.pkl")

        if os.path.exists(path_scaler_features) and os.path.exists(path_scaler_target):
            scaler_features = pickle.load(open(path_scaler_features,"rb"))
            scaler_target = pickle.load(open(path_scaler_target,"rb"))
        else:
            raise FileNotFoundError(f"The feature and target scalers can not be found at the directory below:\n {dir_path}") 
        
        return scaler_features, scaler_target

    @staticmethod
    def parse_glm_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=True, allow_abbrev=False)
        parser.add_argument("--target_distribution_name", default="compound_poisson")

        parser.add_argument("--mu_distribution_name", default="normal")
        parser.add_argument("--mu_link_name", default="identity",help="name of link function used for mu distribution")
        parser.add_argument("--mu_params", default=None, type=tuple_type)

        parser.add_argument("--dispersion_distribution_name", default="gamma")
        parser.add_argument("--dispersion_link_name", default="relu",help="name of link function used for mu distribution")
        parser.add_argument("--disp_params", default=None, type=tuple_type)

        parser.add_argument("--p_link_name", default="sigmoid")
        parser.add_argument("--p_params", default=None, type=tuple_type)



        parser.add_argument("--pos_weight", default=2.0, type=float ,help="The relative weight placed on examples where rain did occur when calculating the loss")

        # Compound Poisson arguments
        parser.add_argument('--cp_version', default=None, type=int)
        parser.add_argument('--max_j', default=None, type=int)
        parser.add_argument('--j_window_size',default=None, type=int)
        parser.add_argument('--target_range',default=(0,4), type=tuple_type)


        glm_args = parser.parse_known_args()[0]
        return glm_args


    @staticmethod
    def get_ckpt_path(_dir_checkpoint, mode='best'):
        if mode == 'best':
            checkpoint_yaml_file = os.path.join(_dir_checkpoint, "best_k_models.yaml")
            # key= ckptpath, value = val_loss
            scores_dict = yaml.load(open(checkpoint_yaml_file, "r"), Loader=yaml.FullLoader)
            best_ckpt_path = min(scores_dict, key=scores_dict.get)

            if os.path.exists(best_ckpt_path) == False: 
                    
                best_ckpt_path = os.path.abspath( os.path.join(
                    Path(_dir_checkpoint).parents[4],
                    best_ckpt_path) )

            path = best_ckpt_path

        else:
            raise NotImplementedError

        return path

MAP_NAME_GLM = {'DGLM':NeuralDGLM}
