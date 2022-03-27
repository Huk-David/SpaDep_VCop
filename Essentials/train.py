from distributions import LIST_PVar_Model
import torch
from torch.nn import functional as F
from torch import nn
import numpy as np
from argparse import ArgumentParser
from glms import NeuralDGLM
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import pickle
import pytorch_lightning as pl
import json
from dataloaders import ToyDataset, AustraliaRainDataset, MAP_NAME_DSET
import torchtyping
import argparse
from neural_nets import MAP_NAME_NEURALMODEL
from glms import MAP_NAME_GLM
import os
import glm_utils
import types

if __name__ == '__main__':

    parent_parser = ArgumentParser(add_help=False, allow_abbrev=False)

    # Train args
    train_parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=True, allow_abbrev=False
            )
    train_parser.add_argument("--gpus", default=1)
    train_parser.add_argument("--sample_size", default=1000)
    train_parser.add_argument("--dataset", default="australia_rain", choices=["toy","australia_rain"])
    train_parser.add_argument("--nn_name", default="HLSTM", choices=["MLP","HLSTM"])
    train_parser.add_argument("--glm_name", default="DGLM", choices=["DGLM"])
    train_parser.add_argument("--max_epochs", default=100, type=int)
    train_parser.add_argument("--debugging",action='store_true' )


    train_args = train_parser.parse_known_args()[0]
    
    # add model specific args
    model_args = MAP_NAME_NEURALMODEL[train_args.nn_name].parse_model_args(parent_parser)
    
    # add data specific args
    data_args = MAP_NAME_DSET[train_args.dataset].parse_data_args(parent_parser)

    # add glm specific args
    glm_args = MAP_NAME_GLM[train_args.glm_name].parse_glm_args(parent_parser)

        
    # Generate Dataset 
    if train_args.dataset == 'toy':

        # Randomly Sample the co-effecients a,b,c
        coeffs = { 'c':torch.rand(data_args.input_shape), 'x':torch.randint(0,3, data_args.input_shape), 'x^2':torch.randint(0,3,data_args.input_shape) }
        target_func = lambda inp: torch.sum( coeffs['c'] + inp*coeffs['x'] + inp*coeffs['x^2'], dim=-1 )
        
        ds_train, ds_val, ds_test, scaler_features, scaler_target = ToyDataset.get_dataset( target_func=target_func, **vars(data_args))
    
    elif train_args.dataset == "australia_rain":
        ds_train, ds_val, ds_test, scaler_features, scaler_target = AustraliaRainDataset.get_dataset(**vars(data_args),
                                                                        target_distribution_name=glm_args.target_distribution_name,
                                                                        target_range=glm_args.target_range)

        data_args.input_shape = ( len( ds_train.datasets[0].features.columns ), )
    else:
        raise NotImplementedError

    # Create the DataLoaders
    dl_train =  DataLoader(ds_train, 240, shuffle=True, num_workers=6, drop_last=False)
    dl_val = DataLoader(ds_train, 1280, shuffle=False, num_workers=4, drop_last=False)
    dl_test = DataLoader(ds_train, 1280, shuffle=False, num_workers=4, drop_last=False)
    
    # Load GLM Model
    glm_class = MAP_NAME_GLM[train_args.glm_name]
    nn_params = { **vars(model_args), **
                    dict( input_shape = data_args.input_shape,
                            output_shape = data_args.output_shape,
                            p_variable_model = glm_args.target_distribution_name in LIST_PVar_Model) }
                    

    glm = glm_class(nn_name=train_args.nn_name, 
                        nn_params = nn_params,
                        scaler_target=scaler_target,
                        scaler_features=scaler_features,
                        **vars(glm_args),
                        min_rain_value=data_args.min_rain_value,
                        debugging=train_args.debugging)

    # Define the trainer
    trainer = pl.Trainer(   gpus=train_args.gpus,
                            default_root_dir = f"Checkpoints/{train_args.glm_name}_{train_args.nn_name}_{glm_args.target_distribution_name}",
                            callbacks =[EarlyStopping(monitor="val_loss/loss", patience=2),
                                            ModelCheckpoint(
                                                monitor="val_loss/loss",
                                                filename='{epoch}-{step}-{val_loss/loss:.3f}-{val_metric/mse_rain:.3f}',
                                                save_last=False,
                                                auto_insert_metric_name=True,
                                                save_top_k=1)
                                             ] ,
                            enable_checkpointing=True,
                            precision=16,
                            max_epochs=train_args.max_epochs)

    # Patching ModelCheckpoint checkpoint name creation
    mc = next( filter( lambda cb: isinstance(cb, ModelCheckpoint), trainer.callbacks) )
    mc._format_checkpoint_name = types.MethodType(glm_utils._format_checkpoint_name, mc)

    # Save the scalers
    glm.save_scalers( trainer.logger.log_dir )

    # Adding debugging to the components of loss function 
    if train_args.debugging and glm_args.target_distribution_name=="compound_poisson":
        glm.loss_fct.tblogger = trainer.logger.experiment


    # Fit the Trainer
    trainer.fit(glm, 
                    train_dataloaders=dl_train,
                    val_dataloaders=dl_val )
    
    # Test the Trainer
    trainer.test(dataloaders=dl_test, ckpt_path='best')


# Training scripts

## Gamma model
## python3 train.py --gpus 1 --dataset australia_rain --nn_name HLSTM --glm_name DGLM --max_epochs 150 --target_distribution_name gamma_hurdle --mu_distribution_name uniform_positive --mu_link_name xshiftn_relu_timesm_yshifteps --mu_params 0.4,6.0 --dispersion_distribution_name uniform_positive --dispersion_link_name xshiftn_relu_timesm_yshifteps --disp_params 0.5,6.0 --pos_weight 1.0 --target_range 0,8 --debugging

## LogNormal
## python3 train.py --gpus 1 --dataset australia_rain --nn_name HLSTM --glm_name DGLM --max_epochs 150 --target_distribution_name lognormal_hurdle --mu_distribution_name uniform_positive --mu_link_name xshiftn_relu_timesm_yshifteps --mu_params 0.5,6.0 --dispersion_distribution_name uniform_positive --dispersion_link_name xshiftn_relu_timesm_yshifteps --disp_params 0.5,6.0 --pos_weight 1.0 --target_range 0,4 --debugging

## Compound Poisson v2
## python3 train.py --gpus 1 dataset australia_rain nn_name HLSTM glm_name DGLM --max_epochs 150 --target_distribution_name compound_poisson --mu_distribution_name uniform_positive --mu_link_name xshiftn_relu_timesm_yshifteps --mu_params 0.1,10.0 --dispersion_distribution_name uniform_positive --dispersion_link_name xshiftn_relu_timesm_yshifteps --disp_params 0.5,6.0 --pos_weight 1 --p_link_name divn_sigmoid_clampeps_yshiftm --p_params 2,1 --cp_version 2 --max_j 12 --target_range 0,2 --debugging

## Compound Poisson v3
## python3 train.py --gpus 1 dataset australia_rain nn_name HLSTM glm_name DGLM --max_epochs 150 --target_distribution_name compound_poisson --mu_distribution_name uniform_positive --mu_link_name xshiftn_relu_timesm_yshifteps --mu_params 0.5,6.0 --dispersion_distribution_name uniform_positive --dispersion_link_name xshiftn_relu_timesm_yshifteps --disp_params 0.5,6.0 --p_link_name divn_sigmoid_clampeps_yshiftm --p_params 2,1 --pos_weight 1.0 --cp_version 3 --max_j 12 --target_range 0,6 --debugging

## Compound Poisson v4
## python3 train.py --gpus 1 dataset australia_rain nn_name HLSTM glm_name DGLM --max_epochs 13 --target_distribution_name compound_poisson --mu_distribution_name uniform_positive --mu_link_name xshiftn_relu_timesm_yshifteps --mu_params 0.1,10.0 --dispersion_distribution_name uniform_positive --dispersion_link_name xshiftn_relu_timesm_yshifteps --disp_params 0.5,6.0 --pos_weight 1.0 --p_link_name divn_sigmoid_clampeps_yshiftm --p_params 2,1 --cp_version 4 --j_window_size 3 --target_range 0,2 --debugging

## Compound Poisson v5
## python3 train.py --gpus 1 dataset australia_rain nn_name HLSTM glm_name DGLM --max_epochs 150 --target_distribution_name compound_poisson --mu_distribution_name uniform_positive --mu_link_name xshiftn_relu_timesm_yshifteps --mu_params 0.5,6.0 --dispersion_distribution_name uniform_positive --dispersion_link_name xshiftn_relu_timesm_yshifteps --disp_params 0.5,6.0 --p_link_name divn_sigmoid_clampeps_yshiftm --p_params 2,1 --pos_weight 1.0 --cp_version 5 --j_window_size 3 --target_range 0,6 --debugging