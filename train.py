from dataloaders import ToyDataset, AustraliaRainDataset,Era5EobsDataset, MAP_NAME_DSET
import torch
from torch.nn import functional as F
from torch import nn
import numpy as np
from argparse import ArgumentParser 
from glms import NeuralDGLM
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

import pytorch_lightning as pl

import os
import torchtyping
import argparse
from neural_nets import MAP_NAME_NEURALMODEL
from glms import MAP_NAME_GLM
import glob
import glm_utils
import types
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.datapipes.iter.combinatorics import ShufflerIterDataPipe
from torch.utils.data.datapipes.iter.utils import IterableWrapperIterDataPipe
from pytorch_lightning.profiler import AdvancedProfiler
from typing import Dict, Any
import yaml
from pytorch_lightning.utilities.cloud_io import get_filesystem
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
from pytorch_lightning.utilities import _OMEGACONF_AVAILABLE, AttributeDict
if _OMEGACONF_AVAILABLE:
    from omegaconf import OmegaConf
    from omegaconf.dictconfig import DictConfig
    from omegaconf.errors import UnsupportedValueType, ValidationError

def load_hparams_from_yaml(config_yaml: str, use_omegaconf: bool = True) -> Dict[str, Any]:
    """Load hparams from a file.

        Args:
            config_yaml: Path to config yaml file
            use_omegaconf: If omegaconf is available and ``use_omegaconf=True``,
                the hparams will be converted to ``DictConfig`` if possible.

    >>> hparams = Namespace(batch_size=32, learning_rate=0.001, data_root='./any/path/here')
    >>> path_yaml = './testing-hparams.yaml'
    >>> save_hparams_to_yaml(path_yaml, hparams)
    >>> hparams_new = load_hparams_from_yaml(path_yaml)
    >>> vars(hparams) == hparams_new
    True
    >>> os.remove(path_yaml)
    """
    fs = get_filesystem(config_yaml)
    if not fs.exists(config_yaml):
        rank_zero_warn(f"Missing Tags: {config_yaml}.", category=RuntimeWarning)
        return {}

    with fs.open(config_yaml, "r") as fp:
        hparams = yaml.unsafe_load(fp)

    if _OMEGACONF_AVAILABLE:
        if use_omegaconf:
            try:
                return OmegaConf.create(hparams)
            except (UnsupportedValueType, ValidationError):
                pass
    return hparams

pl.core.saving.load_hparams_from_yaml = load_hparams_from_yaml

def train( train_args, data_args, glm_args, model_args ):
    # Define the trainer    
    trainer = pl.Trainer(gpus=train_args.gpus,
                        default_root_dir = f"Checkpoints/{train_args.dataset}_{train_args.glm_name}_{train_args.nn_name}_{glm_args.target_distribution_name}",
                        callbacks =[EarlyStopping(monitor="val_loss/loss", patience=7 if data_args.locations!=["All"] else 8 ),
                                        ModelCheckpoint(
                                            monitor="val_loss/loss",
                                            filename='{epoch}-{step}-{val_loss/loss:.3f}-{val_metric/mse_rain:.3f}',
                                            save_last=False,
                                            auto_insert_metric_name=True,
                                            save_top_k=1)
                                            ],
                        enable_checkpointing=True,
                        precision=16,
                        max_epochs=train_args.max_epochs,
                        num_sanity_val_steps=0,
                        limit_train_batches=20 if train_args.debugging else None,
                        limit_val_batches=5 if train_args.debugging else None,
                        limit_test_batches=5 if train_args.debugging else None,
                        val_check_interval=None if train_args.debugging else ( train_args.val_check_interval if train_args.val_check_interval<=1 else int(train_args.val_check_interval) ),
                        # profiler = AdvancedProfiler(
                        #     f"Checkpoints/{train_args.dataset}_{train_args.glm_name}_{train_args.nn_name}_{glm_args.target_distribution_name}",
                        #     "profile.txt") if train_args.debugging else None,
                        
                        )

    # Generate Dataset 
    if train_args.dataset == 'toy':

        # Randomly Sample the co-effecients a,b,c
        coeffs = { 'c':torch.rand(data_args.input_shape), 'x':torch.randint(0,3, data_args.input_shape), 'x^2':torch.randint(0,3,data_args.input_shape) }
        target_func = lambda inp: torch.sum( coeffs['c'] + inp*coeffs['x'] + inp*coeffs['x^2'], dim=-1 )
        
        ds_train, ds_val, ds_test, scaler_features, scaler_target = ToyDataset.get_dataset( target_func=target_func, **vars(data_args))

        cf = default_collate
        shuffle = True
        worker=None

    elif train_args.dataset == "australia_rain":

        ds_train, ds_val, ds_test, scaler_features, scaler_target = AustraliaRainDataset.get_dataset(**vars(data_args),
                                                        target_distribution_name=glm_args.target_distribution_name,
                                                        target_range=glm_args.target_range)

        data_args.input_shape = ( len( ds_train.datasets[0].features.columns ), )

        cf = default_collate
        shuffle = True
        worker_init_fn=None

    elif train_args.dataset == "uk_rain":
        
        ds_train, ds_val, ds_test, scaler_features, scaler_target = Era5EobsDataset.get_dataset( data_args,
                                                                        target_distribution_name=glm_args.target_distribution_name,
                                                                        target_range=glm_args.target_range,
                                                                        gen_size=data_args.gen_size,
                                                                        workers = train_args.workers,
                                                                        trainer_dir=trainer.logger.log_dir )
        cf = glm_utils.default_collate_concat
        worker_init_fn=Era5EobsDataset.worker_init_fn
               
        # Needs to be true for suffling
        assert ds_train.rain_data.data_len_per_location/ds_train.rain_data.lookback == ds_train.mf_data.data_len_per_location/ds_train.mf_data.lookback, "Per location Length of target and feature do not match {ds_train.rain_data.data_data_len_per_locationlen}-{ds_train.mf_data.data_len_per_location}"
        
            # Shuffling Datasetes

        if data_args.shuffle:
            buffer_size = ds_train.loc_count*(4*12) 
            buffer_size = buffer_size if not train_args.debugging else 5
            ds_train_iter = IterableWrapperIterDataPipe(ds_train, deepcopy=False)

            ds_train = ShufflerIterDataPipe(ds_train_iter, 
                            #HERE adjust data length to factor in other locations ont just 1
                            buffer_size=buffer_size, #Each datum represents 7 days long
                            unbatch_level=0)

    else:
        raise NotImplementedError

    # Create the DataLoaders    
    dl_train =  DataLoader(ds_train, train_args.batch_size, shuffle=data_args.shuffle,
                            num_workers=train_args.workers, drop_last=False,
                            pin_memory=True,
                            collate_fn=cf, 
                            worker_init_fn=worker_init_fn,
                            prefetch_factor=train_args.prefetch,
                            persistent_workers=True )

    dl_val = DataLoader(ds_val, train_args.batch_size, shuffle=False, 
                            num_workers=train_args.workers,
                            drop_last=False, collate_fn=cf,
                            pin_memory=True,
                            worker_init_fn=worker_init_fn,
                            prefetch_factor=train_args.prefetch,
                            persistent_workers=True)

    dl_test = DataLoader(ds_test, train_args.batch_size, 
                            shuffle=False,
                            num_workers=train_args.workers,
                            drop_last=False, collate_fn=cf, 
                            pin_memory=True,
                            worker_init_fn=worker_init_fn)
    

    # Load GLM Model
    glm_class = MAP_NAME_GLM[train_args.glm_name]
    nn_params = { **vars(model_args),
                    **dict( input_shape = data_args.input_shape,
                            output_shape = data_args.output_shape,
                            )}
                    
    if train_args.dataset == "uk_rain":
        nn_params['lookback'] = data_args.lookback_target
        nn_params['tfactor'] = data_args.lookback_feature // data_args.lookback_target

    glm = glm_class(nn_name=train_args.nn_name, 
                        nn_params = nn_params,
                        scaler_target=scaler_target,
                        scaler_features=scaler_features,
                        **vars(glm_args),
                        min_rain_value=data_args.min_rain_value,
                        task = train_args.dataset,
                        debugging=train_args.debugging,
                        dconfig= data_args)

    # Patching ModelCheckpoint checkpoint name creation
    mc = next( filter( lambda cb: isinstance(cb, ModelCheckpoint), trainer.callbacks) )
    mc._format_checkpoint_name = types.MethodType(glm_utils._format_checkpoint_name, mc)

    if not train_args.test_version:
        # Save the scalers
        glm.save_scalers( trainer.logger.log_dir, dconfig=data_args, glmconfig=glm_args )

        # Fit the Trainer
        try:
            trainer.fit(
                        glm, 
                        train_dataloaders=dl_train,
                        val_dataloaders=dl_val
                         )
        except RuntimeError as e:
            print("e")
            raise e

    # Test the Trainer
    trainer.test(dataloaders=dl_test, ckpt_path='best')

def test( train_args, data_args, glm_args ):

    _dir  = os.path.join(
        f"Checkpoints/{train_args.dataset}_{train_args.glm_name}_{train_args.nn_name}_{glm_args.target_distribution_name}",
        "lightning_logs",f"version_{train_args.test_version}")

    hparams_path = os.path.join( _dir, "hparams.yaml")
    checkpoint_path = next( ( elem for elem in glob.glob(os.path.join( _dir, "checkpoints", "*")) 
                                if elem[-4:]=="ckpt"))
    scaler_features, scaler_target = NeuralDGLM.load_scalers( os.path.join( _dir) )

    #Loading state
    glm_class = MAP_NAME_GLM[train_args.glm_name]
    glm = glm_class.load_from_checkpoint(checkpoint_path, hparams_file=hparams_path, 
                        scaler_features=scaler_features,
                        scaler_target=scaler_target,
                        save_hparams=False)
    
    trainer = pl.Trainer(resume_from_checkpoint=checkpoint_path, 
                        precision=16,
                        enable_checkpointing=not train_args.debugging,
                        logger=False,
                        gpus=train_args.gpus,
                        default_root_dir = f"Checkpoints/{train_args.dataset}_{train_args.glm_name}_{train_args.nn_name}_{glm_args.target_distribution_name}")

    # Making Dset
    if train_args.dataset == "uk_rain":
        ds_test = Era5EobsDataset.get_test_dataset( data_args,
                                                    target_distribution_name=glm_args.target_distribution_name,
                                                    target_range=glm_args.target_range,
                                                    scaler_features=scaler_features,
                                                    scaler_target=scaler_target,
                                                    gen_size=data_args.gen_size,
                                                    workers = train_args.workers)
    
        dl_test = DataLoader(ds_test, train_args.batch_size, 
                                shuffle=False,
                                num_workers=train_args.workers,
                                drop_last=False, collate_fn=glm_utils.default_collate_concat, 
                                worker_init_fn=Era5EobsDataset.worker_init_fn)

    # Patching ModelCheckpoint checkpoint name creation


    # Test the Trainer
    trainer.test(glm, dataloaders=dl_test)

if __name__ == '__main__':

    parent_parser = ArgumentParser(add_help=False, allow_abbrev=False)

    # Train args
    train_parser = argparse.ArgumentParser(parents=[parent_parser], add_help=True, allow_abbrev=False)
            
    train_parser.add_argument("--gpus", default=1)
    train_parser.add_argument("--sample_size", default=1000)
    train_parser.add_argument("--dataset", default="australia_rain", choices=["toy","australia_rain","uk_rain"])

    train_parser.add_argument("--nn_name", default="HLSTM", choices=["MLP","HLSTM","HLSTM_tdscale", "HConvLSTM_tdscale"])
    train_parser.add_argument("--glm_name", default="DGLM", choices=["DGLM"])
    train_parser.add_argument("--max_epochs", default=500, type=int)
    train_parser.add_argument("--batch_size", default=16, type=int)
    train_parser.add_argument("--debugging",action='store_true' )
    train_parser.add_argument("--workers",default=8, type=int )
    train_parser.add_argument("--test_version",default=None, type=int, required=False ) #TODO: implelment logic such that a trained model can just be tested on a given dataset
    train_parser.add_argument("--val_check_interval", default=1.0, type=float)
    train_parser.add_argument("--prefetch",type=int, default=4, help="Number of batches to prefetch" )
    train_parser.add_argument("--hypertune",type=bool, default=False)
    
    train_args = train_parser.parse_known_args()[0]
    
    # add model specific args
    model_args = MAP_NAME_NEURALMODEL[train_args.nn_name].parse_model_args(parent_parser)
    
    # add data specific args
    data_args = MAP_NAME_DSET[train_args.dataset].parse_data_args(parent_parser)

    # add glm specific args
    glm_args = MAP_NAME_GLM[train_args.glm_name].parse_glm_args(parent_parser)
    
    if train_args.test_version==None:
        train(train_args, data_args, glm_args, model_args)
    else:
        test( train_args, data_args, glm_args )
    

    




