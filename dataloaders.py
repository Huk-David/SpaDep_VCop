from genericpath import exists
from re import L, M
from tokenize import String
from attr import has
from netCDF4 import Dataset as nDataset
#from aiohttp import worker --- removed #
from functools import lru_cache
import numpy as np
import pandas as pd
from pandas._libs import missing
import torch
from torch._C import Value
import torch.distributions as td
from torch.functional import Tensor
from torch.utils.data import Dataset, Subset
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler, FunctionTransformer, MaxAbsScaler   
import math
import copy
# from pytorch_lightning.profiler import Profiler, PassThroughProfiler ---- removed #
import os
import ujson
import pickle
import regex  as re
from typing import Tuple, Callable,  Union, Dict, List, TypeVar
from torchtyping import TensorDetail,TensorType
import argparse
import json
import ujson
import xarray as xr
import itertools as it
import glob
from torch.utils.data.datapipes.iter.combinatorics import ShufflerIterDataPipe
from torch.utils.data import IterableDataset
import random
from collections import defaultdict
from glm_utils import tuple_type
#from torch.utils.data.datapipes.datapipe import _IterDataPipeSerializationWrapper ---- removed #
"""
    dataloaders.py provides functionality for loading in the following Datasets:

    ToyDataset:
        This is a toy dataset. It can be used to investigate how well a bayesian neural net can model uncertainty.
        A bayesian neural network can learn to model the following two forms of uncertainty:
            1) Uncertainty due to not enough data provided for a particular set of X. e.g. 
                when we have a imbalanced/skewed dataset that has few pairs (X,Y) for some subset of X
            2) Uncertainty due to the dataset being 

    AustraliaDataset
        This a weather dataset. The target is daily rainfall. The input is weather related variables. 
        More info in class description.
    
"""



class ToyDataset(Dataset):
    """
        ToyDataset:
            This is a toy dataset. It can be used to investigate how well a bayesian neural net can model uncertainty.
            A bayesian neural network can learn to model the following two forms of uncertainty:
                1) Uncertainty due to not enough data provided for a particular set of X. e.g. 
                    when we have a imbalanced/skewed dataset that has few pairs (X,Y) for some subset of X
                2) Uncertainty due to the dataset being 
        
        This class takes features and target and creates a torch Dataset class
    """
    def __init__(self, features, target ):
        self.features = features
        self.target = target

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        target = self.target[idx]
        return feature, target
    
    @staticmethod
    def get_distribution(self, distribution_name ,**distribution_kwargs ):

        MAP_DISTR_SAMPLE = {
            'uniform': lambda lb, ub: td.Uniform(lb, ub),
            
            'mv_uniform': lambda lb=0, ub=1:td.Independent( td.Uniform(lb, ub), 1 ),

            'mv_normal': lambda loc=torch.zeros((6,)), covariance_matrix=torch.eye(6): td.MultivariateNormal( loc , covariance_matrix ),
            
            'mv_lognormal': lambda loc=torch.zeros((6,)), scale=torch.ones( (6,) ):td.Independent( td.LogNormal( loc , scale ), 1 )

        }

        distr =  MAP_DISTR_SAMPLE[distribution_name](**distribution_kwargs)

    @staticmethod
    def get_dataset(    
        input_shape:Tuple[int,...],
        target_func:Callable[ [float, float, float], float],
        sample_size:int=1000,
        distribution_name:str='uniform',
        distribution_kwargs:dict={},
        noise_method:str='random_guassian',
        noise_sample_params:dict={ 'loc':0, 'scale':0.1 },
        **kwargs):
        raise NotImplementedError
        assert noise_method in [ 'random_guassian','increasing_at_extremes', 'increasing_at_maximum', 'increasing_at_minimum' ,'intervals']
        
        """Generates 3 Dataset objects for the training, validation and testing steps.
            The toy dataset generated will contain (X,Y) where Y = c + ax + bx^2 + epsilon, where epsilon is random noise.
            The co-effecients a,b,c are chosen at random.
            In the actual implementation we create Y=c + ax + bx^2, then we pair Y with (x+epsilon). This allows variance to scale with size of Y

        Args:
            input_shape (Tuple[int,...]): A tuple representing the input shape of the data, excluding the batch dimension.
            target_func (Callable[ [float, float], float]): The function mapping X -> Y - epsilon.
            sample_size (int): An int explaining the size of the combined size of the train, validation and test set.
            distribution_name (str): Distribution from which to sample X
            distribution_kwargs (dict): Dictionary containing key-word params for the sampling of X from the class relating to distribution_name
            noise_method (str): The method used to sample the noise term, epsilon 

        Returns:
            Dataset: Returns the Dataset object which produces paired samples of features and target data
        """
        
        distr =  ToyDataset.get_distribution( distribution_name, **distribution_kwargs)

        X = distr.sample( (sample_size,) )
        #add noise to X instead of Y
        X_pertubed = ToyDataset.add_noise(X, noise_method, noise_sample_params) 

        Y = target_func( X_pertubed )

        if Y.ndim==1:
            Y = Y.unsqueeze(-1)

        ds = ToyDataset(X,Y)

        # splitting dataset into train, validation and test
        train_idx_start = 0
        val_idx_start = int(0.6*sample_size)
        test_idx_start = int(0.8*sample_size)

        # Dividing into train, test, val
        ds_train = Subset(ds, indices = list( range(train_idx_start,val_idx_start) ) ) 
        ds_val = Subset(ds, indices = list( range(val_idx_start, test_idx_start) ))
        ds_test = Subset(ds, indices = list( range(val_idx_start, test_idx_start) ))

        return ds_train, ds_val, ds_test

    @staticmethod
    def add_noise( input:torch.Tensor,
        method:str='random_guassian',
        noise_sample_params:dict=None) -> torch.Tensor: 
        """This method adds random pertubations (noise) to an input tensor

        Args:
            input (Tensor): [Tensor] Tensor to add noise to
            method (str, optional): [description]. Defaults to 'increasing_at_extremes'.
            noise_sample_params (dict, optional): Dictionary containing keyword arguments
                for distribution used to sample noise. Defaults to None.

        Raises:
            NotImplementedError: Only implemented noise method is "random_guassian"
            ValueError: [description]

        Returns:
            [Tensor]: input + noise
        """


        if method == 'random_guassian':
            input = input + td.Normal(**noise_sample_params ).sample( tuple(input.shape) ) 
            input.clamp_min(0.00)

        # Add noise proportional to decile the data is in
        elif method == 'increasing_at_extremes':
            raise NotImplementedError

        # Add relatively more noise to the max deciles
        elif method == 'increasing_at_maximum':
            raise NotImplementedError

        # Add relatively more noise to the minimum deciles
        elif method == 'increasing_at_minimum':
            raise NotImplementedError

        # Add more noise proportional to the size of the value
        elif method == 'intervals':
            raise NotImplementedError

        else:
            raise ValueError
        
        return input

    @staticmethod
    def parse_data_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=True, allow_abbrev=False)
        parser.add_argument("--target_params", default=[0.0, 3.0, 0.5], type=lambda _str: json.loads(_str) )

        parser.add_argument("--input_shape", default=(1,) )
        parser.add_argument("--output_shape", default=(1,) )
        parser.add_argument("--noise_sample_params", default={ 'loc':0, 'scale':0.1 }, type=dict)
        data_args = parser.parse_known_args()[0]
        return data_args

class AustraliaRainDataset(Dataset):
    """
        Dataset source: https://www.kaggle.com/fredericods/forecasting-rain/data?select=weatherAUS.csv

        This dataset contains about 10 years of daily weather observations from numerous Australian weather stations.
        
        This dataset provides point estimates.

        The target RainTomorrow means: Did it rain the next day? Yes or No.

        Note: You should exclude the variable Risk-MM when training your binary classification model. If you don't exclude it, you will leak the answers to your model and reduce its predictability. Read more about it here.

    """

    # list of locations that can be used for the Australia Dataset
    locations = sorted(['Albury', 'BadgerysCreek', 'Cobar', 'CoffsHarbour', 'Moree',
            'Newcastle', 'NorahHead', 'NorfolkIsland', 'Penrith', 'Richmond',
            'Sydney', 'SydneyAirport', 'WaggaWagga', 'Williamtown',
            'Wollongong', 'Canberra', 'Tuggeranong', 'MountGinini', 'Ballarat',
            'Nhil', 'Portland', 'Watsonia', 'Dartmoor', 'Brisbane', 'Cairns',
            'GoldCoast', 'Townsville', 'Adelaide', 'MountGambier', 'Nuriootpa',
            'Woomera', 'Albany', 'Witchcliffe', 'PearceRAAF', 'PerthAirport',
            'Perth', 'SalmonGums', 'Walpole', 'Hobart', 'Launceston',
            'AliceSprings', 'Darwin', 'Katherine', 'Uluru'])
    
    valid_locations = sorted(['Adelaide', 'Albury', 'AliceSprings', 'BadgerysCreek', 'Ballarat',
       'Brisbane', 'Cairns', 'Canberra', 'Cobar', 'CoffsHarbour',
       'Dartmoor', 'Darwin', 'GoldCoast', 'Hobart',
       'Moree', 'MountGambier', 'Nhil', 'NorahHead',
       'NorfolkIsland', 'Nuriootpa', 'PearceRAAF', 'Perth',
       'PerthAirport', 'Portland', 'Richmond', 'Sydney', 'SydneyAirport',
       'Townsville', 'Tuggeranong', 'Uluru', 'WaggaWagga', 'Walpole',
       'Watsonia', 'Williamtown', 'Witchcliffe', 'Wollongong', 'Woomera']) # These are the locations which have enough valid entries to not cause errors with default settings
    
    def __init__(self,
                    features:pd.DataFrame,
                    targets:pd.DataFrame,
                    lookback:int=7,
                    location:str=None) -> None:
        """Initializes at Australian Rain Dataset.
            Expected use of this dataset: Predict rainfall at, using weather information from day
            Due to erroneous information at specific dates in feature dataset,
                there are some days which have been dropped from feature and target datasets. However,
                this Dataset expects a sequence of days to be used for generating a loss. Therefore, we must
                ignore any l=lookback period where the period contains any days that have been dropped from the 
                dataset.
                

        Args:
            features (pd.DataFrame): [A pandas DataFrame containing the features. The index must be a datetime index ]
            targets (pd.DataFrame): [A pandas DataFrame containing the targets. The index must be a datetime index ]
            lookback (int, optional): Lookback Period
            location (str, optional): City name for which the data is from. Defaults to None.
        """
        super().__init__()
        
        self.features = features
        self.targets = targets
        self.lookback = lookback
        self.location = location

        assert self.features.index.equals(self.targets.index), "Datetime Index of features and target are not the same"
        
        self.create_index_exclude_missing_days()

    def create_index_exclude_missing_days(self) -> None:
        """
            This function creates a new list. This list contains the indicies of days in the feature and target,
            which are valid for use as an input and target.

            A day d, is valid if all the days in the the range d-lookback to d are included in the feature and target dataset.
            
        """        
        self.dates = copy.deepcopy(self.features.index)
        self.indexes_filtrd = list(range(len(self.dates)))

        # A list of days missing from the index
        missing_days = pd.date_range(start=self.features.index[0], end=self.features.index[-1]).difference(self.features.index)

        # Let m_day be a day missing from the feature and target dataframes
        # For each m_day, get the list of days which need m_day for prediction
            # Then remove this list of days from the dates
        for m_day in reversed(missing_days):
            
            li_affected_day = pd.date_range(start = m_day, end = m_day +  pd.DateOffset(days=self.lookback) )
                
            for affected_day in reversed(li_affected_day):
                
                if affected_day in self.dates:

                    index_affected_day = self.dates.get_loc(affected_day)

                    if type(index_affected_day)==slice:
                        raise ValueError
                        
                    elif type(index_affected_day)==int:
                        self.dates = self.dates.drop(affected_day)
                        self.indexes_filtrd.pop(index_affected_day)

                    else:
                        pass
        
    def __len__(self):
        return max(0,len(self.indexes_filtrd) - self.lookback)
        
    def __getitem__(self, index):
        
        index = index + self.lookback
        adj_index = self.indexes_filtrd[index]
        features = self.features.iloc[ adj_index-self.lookback:adj_index ].to_numpy(dtype=np.float32)
        targets = self.targets.iloc[ adj_index-self.lookback:adj_index].to_numpy()
        
        return torch.tensor(features), torch.tensor(targets)

    @staticmethod
    def calculate_velocity(dict_winddirection_radians:Dict[str,float], wind_direction, wind_speed ) -> float:
        """Maps a wind direction and wind speed to a wind velocity

        Args:
            dict_winddirection_radians ([type]): A dictionary containing a mapping from wind direction (str) to radians
            wind_direction (str): A str such as N or SE explaining the wind direction
            wind_speed (float): 

        Returns:
            (str): wind velocity
        """
        assert ( wind_direction in list(dict_winddirection_radians.keys()) ) or np.isnan(wind_direction)
        
        radians = dict_winddirection_radians[wind_direction]
        return wind_speed*math.cos(radians), wind_speed*math.sin(radians)

    @staticmethod
    def get_dataset(start_date:str="2008-12-01", end_date:str="2021-07-03",
                    locations=['Albury'], lookback=6,
                    train_val_test_split:list = [0.6,0.2,0.2],
                    target_distribution_name:str="lognormal_hurdle", 
                    target_range=(0,4),
                    **kwargs ) -> Tuple:
        """Creates a train, test and validation dataset for Australian Rain.

        Args:
            start_date (str, optional): [starting date, in format YYYY-MM-DD]. Defaults to "2008-12-01".
            end_date (str, optional): [ending data, in format YYYY-MM-DD]. Defaults to "2021-07-03".
            locations (list[str], optional): [List of locations to use in datasets]. Defaults to ['Albury'].
            lookback (int, optional): [lookback to use]. Defaults to 6.
            train_val_test_split (list, optional): Proportion of dataset to use for train/val/test set. Defaults to [0.6,0.2,0.2].
            target_distribution_name (str, optional): [Distribution of target variable. This is used for ]. Defaults to "lognormal_hurdle".

        Returns:
            [tuple(Dataset, Dataset, Dataset, MinMaxScaler, None)]: train_dataset, val_dataset, test_dataset, features_scaler, target_scaler 
        
        Note: This class caches the datasets that it creates. Each dataset is unique in its combination of
                'start_date','end_date','locations','lookback','train_val_test_split','target_distribution_name'
        """
        locations = sorted(locations)
        assert all( loc in AustraliaRainDataset.locations for loc in locations), "Invalid Location chosen"
        if len(locations) == 0:
            locations = AustraliaRainDataset.locations
        
        # Retreiving record of previously created datasets
        premade_dset_path = os.path.join('Data','australia_rain','premade_dset_record.txt')
        if os.path.exists(premade_dset_path):
            premade_dsets = pd.read_csv( premade_dset_path)
        else:
            premade_dsets = pd.DataFrame( columns=['path','start_date','end_date','locations','lookback','train_val_test_split','target_distribution_name','target_range'] )

        # Query for if existing dataset is made
        query_res = premade_dsets.query( f"start_date == '{start_date}' and end_date == '{end_date}' and \
                locations == '{ujson.dumps(locations)}' and \
                lookback == {str(lookback)} and \
                train_val_test_split == '{ujson.dumps(train_val_test_split)}' and \
                target_distribution_name == '{target_distribution_name}' and \
                target_range == '{target_range}'" )

        if len(query_res)!=0:

            with open(query_res['path'].iloc[0], "rb") as f:
                pkl_dset_dict = pickle.load( f ) 
            
            concat_dset_train = pkl_dset_dict['concat_dset_train']
            concat_dset_val = pkl_dset_dict['concat_dset_val']
            concat_dset_test= pkl_dset_dict['concat_dset_test']
            scaler_features = pkl_dset_dict['scaler_features']
            scaler_target  = pkl_dset_dict['scaler_target']

        else: 
            # Make dataset from scratch
            data = pd.read_csv("./Data/australia_rain/weatherAUS.csv")

            # Adding Month and Day column
            data.insert(loc=1, column='Month', value = data['Date'].apply(lambda x: x[5:7])) #create column "Month"
            data.insert(loc=2, column='Day', value = data['Date'].apply(lambda x: x[7:10])) #create column

            # Filtering dates included in datasets
            data.Date = pd.to_datetime(data.Date)
            data = data.loc[ (data.Date >=pd.Timestamp(start_date) ) & ( data.Date <= pd.Timestamp(end_date) ) ]
            
            # Adding Season feature
            data.insert(loc=3, column='Season', value = data['Month'].replace(['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'], ['summer','summer', 'summer', 'fall', 'fall', 'fall', 'winter', 'winter', 'winter', 'spring', 'spring', 'spring'])) #create column "Season"
            
            # Converting WindGust into a vector
            # Full list of Directions -> (N, NNE, NE, ENE, E, ESE, SE, SSE, S, SSw, SW, WSW, W, WWN, NW, NNW, N )
            dict_winddirection_radians = {
                'N':math.pi/2, 'NNE': math.pi/3 , 'NE':math.pi/4, 'ENE':math.pi/6, 'E':0, 'ESE':math.pi*(11/6), 'SE':math.pi*(7/4), "SSE":math.pi*(5/3), 
                'S':math.pi*(3/2), 'SSW':math.pi*(4/3), 'SW':math.pi*(5/4), 'WSW':math.pi*(7/6), 'W':math.pi, 'WNW':math.pi*(5/6), 'NW':math.pi*(3/4), 'NNW':math.pi*(2/3),
                np.nan:0
            }
            data[ ['WindGustU', 'WindGustV'] ] = data.apply( lambda row: AustraliaRainDataset.calculate_velocity( dict_winddirection_radians, row['WindGustDir'], row['WindGustSpeed'] ), axis=1, result_type='expand' )
            data[['WindVelocity9amU', 'WindVelocity9amV']] = data.apply( lambda row: AustraliaRainDataset.calculate_velocity(dict_winddirection_radians, row['WindDir9am'], row['WindSpeed9am'] ), axis=1, result_type='expand' )
            data[['WindVelocity3pmU', 'WindVelocity3pmV']] = data.apply( lambda row: AustraliaRainDataset.calculate_velocity(dict_winddirection_radians, row['WindDir3pm'], row['WindSpeed3pm'] ), axis=1., result_type='expand' )

            # For each location, sorting values by date and stacking in alphabetical order  
            li_dsets = []
            for loc in locations:
                dataset_loc = data[data.Location == loc ]
                dataset_loc = dataset_loc.sort_values(by='Date')
                li_dsets.append(dataset_loc)
            
            data = pd.concat(li_dsets)
            data = data.reset_index(drop=True)

            # Drop low quality columns
            # # The variables Sunshine, Evaporation, Cloud3pm, Cloud9am were removed because they had a low fill percentage
            # # Location was removed, since we want forecast rain regardless the location.
            # # Date, Month, Day and were removed, because Season is going to be used instead.
            # # RISK_MM was removed to avoid data leakage.
            # # Rainfall and RainTomorrow are removed to replicate TRUNET settings
            # # WindGustDir, WindGustSpeed, WindDir9am, WindSpeed9am, WindDir3pm, WindSpeed3pm are dropped since they have been replace continous velocity
            data.set_index(['Date'], inplace=True, drop=True)

            data_final_variables = data.drop(columns=['Sunshine','Evaporation','Cloud3pm','Cloud9am', 'Month', 'Day', 'RISK_MM','RainTomorrow',
                                                        'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindSpeed9am', 'WindDir3pm', 'WindSpeed3pm'],axis=1)
            
            data_final_variables = data_final_variables.dropna(axis=0)

            targets_raw = data_final_variables[['RainToday','Rainfall','Location']]
            features_raw = data_final_variables.drop(columns = ['RainToday','Rainfall']) 
            

            # Scaling Features
            #:Ensure scaler features is only trained on training set
            scaler_features = StandardScaler()

            types_aux = pd.DataFrame(features_raw.dtypes)
            types_aux.reset_index(level=0, inplace=True)
            types_aux.columns = ['Variable','Type']
            numerical_feature = list(types_aux[types_aux['Type'] == 'float64']['Variable'].values)

            features_minmax_transform = pd.DataFrame(data = features_raw)

            # training scaler only on train set            
            end_train_date = pd.Timestamp(start_date) + pd.DateOffset(days=train_val_test_split[0]*(  pd.Timestamp(end_date) - pd.Timestamp(start_date) ).days) 
            _ = features_raw[numerical_feature][ (features_raw[numerical_feature].index >=pd.Timestamp(start_date) ) &
                                                      (features_raw[numerical_feature].index <= end_train_date) ]
            scaler_features.fit( _ )
            features_minmax_transform[numerical_feature] = scaler_features.transform(features_raw[numerical_feature])

                # One Hot Encoding non numeric columns
            location_column = features_minmax_transform['Location']
            features_minmax_transform = features_minmax_transform.drop(columns=['Location'],axis=1)
            features_final = pd.get_dummies(features_minmax_transform)
            features_final['Location'] = location_column
            
            # Scaling Targets
                # Scaling methodology is determined by the target distribution. 
                # Note, specific distributions such as lognormal can not be scaled to 0,1 since they are not invariant under affine transformation 
            scaler_target = MinMaxScaler( feature_range = target_range )

            types_aux = pd.DataFrame(targets_raw.dtypes) 
            types_aux.reset_index(level=0, inplace=True)
            types_aux.columns = ['Variable','Type']
            numerical_target = list(types_aux[types_aux['Type'] == 'float64']['Variable'].values)
            target_transform = pd.DataFrame(data = targets_raw )
            
            if scaler_target is not None:
                _ = targets_raw[numerical_target][ (targets_raw[numerical_target].index >= pd.Timestamp(start_date) ) &
                                                      (targets_raw[numerical_target].index <= end_train_date) ]
                scaler_target.fit( _ )

                target_transform[numerical_target] = scaler_target.transform(targets_raw[numerical_target])

            # replace "Yes","No" with binary
            target_transform['RainToday'] = target_transform['RainToday'].replace(['Yes', 'No'], [1,0])
            targets_final = target_transform

            # Creating seperate datasets for each location
            li_dsets_train = []
            li_dsets_val   = []
            li_dsets_test  = []

            # For each location Creating, the train, validation
            for loc in locations:
                
                X_loc = features_final[ features_final.Location == loc ]
                Y_loc = targets_final[ targets_final.Location == loc ]
                
                start_train_date = pd.Timestamp(start_date)
                end_train_date = start_val_date = start_train_date + pd.DateOffset(days=train_val_test_split[0]*(  pd.Timestamp(end_date) - pd.Timestamp(start_date) ).days) 
                end_val_date = start_test_date = start_val_date + pd.DateOffset(days=train_val_test_split[1]*(  pd.Timestamp(end_date) - pd.Timestamp(start_date) ).days )
                end_test_date = pd.Timestamp(end_date)

                # Date filtering
                X_train = X_loc.loc[ (X_loc.index >= start_train_date) & (X_loc.index <= end_train_date) ]
                Y_train = Y_loc.loc[ (Y_loc.index >= start_train_date) & (Y_loc.index <= end_train_date) ]

                X_val = X_loc.loc[ (X_loc.index >= start_val_date) & (X_loc.index <= end_val_date) ]
                Y_val = Y_loc.loc[ (Y_loc.index >= start_val_date) & (Y_loc.index <= end_val_date) ]

                X_test = X_loc.loc[ (X_loc.index >= start_test_date) & (X_loc.index <= end_test_date) ]
                Y_test = Y_loc.loc[ (Y_loc.index >= start_test_date) & (Y_loc.index <= end_test_date) ]

                # Dropping location column
                X_train = X_train.drop(axis=1, labels=['Location'])
                Y_train = Y_train.drop(axis=1, labels=['Location'])
                X_val = X_val.drop(axis=1, labels=['Location'])
                Y_val = Y_val.drop(axis=1, labels=['Location'])
                X_test = X_test.drop(axis=1, labels=['Location'])
                Y_test = Y_test.drop(axis=1, labels=['Location'])

                if len(X_train)>0:
                    dset_train = AustraliaRainDataset(X_train, Y_train, lookback, loc)
                    li_dsets_train.append(dset_train)
                
                if len(X_val)>0:
                    dset_val = AustraliaRainDataset(X_val, Y_val, lookback, loc)
                    li_dsets_val.append(dset_val)
                
                if len(X_test)>0:
                    dset_test = AustraliaRainDataset(X_test, Y_test, lookback, loc)
                    li_dsets_test.append(dset_test)

            concat_dset_train = torch.utils.data.ConcatDataset(li_dsets_train)
            concat_dset_val = torch.utils.data.ConcatDataset(li_dsets_val)
            concat_dset_test = torch.utils.data.ConcatDataset(li_dsets_test)

            # Caching Dataset 
            ## Create a number for this dataset
            try:
                new_dset_number = int( max( [ re.findall("(?<=/)\d+(?=.pkl)",path_)[0] for path_ in premade_dsets['path'].tolist() ], key=int) ) + 1
            except ValueError:
                new_dset_number = 0

            os.makedirs(os.path.join('Data','australia_rain','premade_dsets'),exist_ok=True)
            path_ = os.path.join('Data','australia_rain','premade_dsets', f'{str(new_dset_number)}.pkl')

            with open(path_,"wb") as f:
                pickle.dump({'concat_dset_train':concat_dset_train, 'concat_dset_val':concat_dset_val, 'concat_dset_test':concat_dset_test, 'scaler_features':scaler_features, 'scaler_target':scaler_target }, f  )
            premade_dsets = premade_dsets.append( {'path':path_,
                                    'start_date':start_date,
                                    'end_date':end_date,
                                    'locations':ujson.dumps(locations),
                                    'lookback':str(lookback),
                                    'train_val_test_split':ujson.dumps(train_val_test_split),
                                    'target_distribution_name': target_distribution_name,
                                    'target_range':str(scaler_target.feature_range)
                                     } , ignore_index=True)
            premade_dsets.to_csv(premade_dset_path, index=False)

        return concat_dset_train, concat_dset_val, concat_dset_test, scaler_features, scaler_target
    
    @staticmethod
    def parse_data_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=True, allow_abbrev=False)

        parser.add_argument("--input_shape", default=(19,) )
        parser.add_argument("--output_shape", default=(1,) )
        parser.add_argument("--lookback", default=7, type=int )
        parser.add_argument("--locations", type= lambda _str:json.loads(_str), default=AustraliaRainDataset.valid_locations )
        parser.add_argument("--min_rain_value",type=float, default=1.0)
        
        data_args = parser.parse_known_args()[0]
        return data_args

# region -- Era5_Eobs
class Generator():
    """
        Base class for Generator classes
        Example of how to use:
            fn = "Data/Rain_Data/rr_ens_mean_0.1deg_reg_v20.0e_197901-201907_djf_uk.nc"
            rain_gen = Generator_rain(fn, all_at_once=True)
            datum = next(iter(grib_gen))
    """

    city_latlon = {
            "London": [51.5074, -0.1278],
            "Cardiff": [51.4816 + 0.15, -3.1791 -0.05], #1st Rainiest
            "Glasgow": [55.8642,  -4.2518], #3rd rainiest
            "Lancaster":[54.466, -2.8007], #2nd hieghest
            "Bradford": [53.7960, -1.7594], #3rd highest
            "Manchester":[53.4808, -2.2426], #15th rainiest
            "Birmingham":[52.4862, -1.8904], #25th
            "Liverpool":[53.4084 , -2.9916 +0.1 ], #18th rainiest
            "Leeds":[ 53.8008, -1.5491 ], #8th
            "Edinburgh": [55.9533, -3.1883],
            "Belfast": [54.5973, -5.9301], #25
            "Dublin": [53.3498, -6.2603],
            "LakeDistrict":[54.4500,-3.100],
            "Newry":[54.1751, -6.3402],
            "Preston":[53.7632, -2.7031 ],
            "Truro":[50.2632, -5.0510],
            "Bangor":[54.2274 - 0, -4.1293 - 0.3],
            "Plymouth":[50.3755 + 0.1, -4.1427],
            "Norwich": [52.6309, 1.2974],
            "StDavids":[51.8812+0.05, -5.2660+0.05] ,
            "Swansea":[51.6214+0.05,-3.9436],
            "Lisburn":[54.5162,-6.058],
            "Salford":[53.4875, -2.2901],
            "Aberdeen":[57.1497,-2.0943-0.05],
            "Stirling":[56.1165, -3.9369],
            "Hull":[53.7676+0.05, 0.3274],
            "Armagh":[54.3503, -6.66528],
            "Bath":[51.380001,-2.360000],
            "Brighton":[50.827778,-0.152778],
            "Cambridge":[52.205276, 0.119167],
            "Canterbury":[51.279999,1.080000],
            "Chelmsford":[51.736099,0.479800],
            "Chester":[53.189999,-2.890000],
            "Coventry":[52.408054, -1.510556],
            "Derby":[52.916668,-1.466667],
            "Exeter":[50.716667,-3.533333],
            "Perth":[56.396999, -3.437000],
            "Sunderland":[54.906101,-1.381130],
            "Wolverhampton":[52.591370,-2.110748],
            "Worcester":[	52.192001,-2.220000],
            "York":[53.958332,-1.080278],
            }

        # The list of boundaries to remove assumes that we are using a 16by16 outer grid with a 4by4 inner grid
    
    invalid_points = {
            0:list(range(0,48+1))+list(range(64,140+1)), 1:list(range(0,48+1))+list(range(64,140+1)), 2:list(range(0,48+1))+list(range(64,140+1)), 3:list(range(0,48+1))+list(range(64,140+1)), 4:list(range(0,48+1))+list(range(64,140+1)), 5:list(range(0,48+1))+list(range(64,140+1)), 
            
            #6-9
            6:list(range(0,48+1))+list(range(64,140+1)), 7:list(range(0,48+1))+list(range(64,140+1)), 8:list(range(0,48+1))+list(range(64,140+1)), 9:list(range(0,48+1))+list(range(64,140+1)),

            #10-13
            10: list(range(0,48+1))+list(range(96,140+1)), 11: list(range(0,48+1))+list(range(76,140+1)), 12 :list(range(0,48+1))+list(range(76,140+1)), 13 :list(range(0,48+1))+list(range(76,140+1)),

            #14-17
            14:list(range(0,48+1))+list(range(96,140+1)), 15:list(range(0,48+1))+list(range(96,140+1)), 16:list(range(0,48+1))+list(range(96,140+1)), 17:list(range(0,48+1))+list(range(96,140+1)),

            #18-21
            18:list(range(0,48+1))+list(range(96,140+1)), 19:list(range(0,48+1))+list(range(96,140+1)), 20:list(range(0,48+1))+list(range(96,140+1)), 21:list(range(0,48+1))+list(range(96,140+1)),

            #22-25
            22:list(range(0,48+1))+list(range(96,140+1)), 23:list(range(0,48+1))+list(range(96,140+1)), 24:list(range(0,48+1))+list(range(96,140+1)), 25:list(range(0,48+1))+list(range(96,140+1)),

            #26-29
            26:list(range(0,48+1))+list(range(96,140+1)), 27:list(range(0,48+1))+list(range(96,140+1)), 28:list(range(0,48+1))+list(range(96,140+1)), 29:list(range(0,48+1))+list(range(96,140+1)),

            #30-33
            30:list(range(0,48+1))+list(range(100,140+1)), 31:list(range(0,48+1))+list(range(100,140+1)), 32:list(range(0,48+1))+list(range(100,140+1)), 33:list(range(0,48+1))+list(range(100,140+1)),

            #34-37
            34:list(range(100,140+1)), 35:list(range(100,140+1)), 36:list(range(100,140+1)), 37:list(range(100,140+1)),

            #38-41
            38:list(range(104,140+1)), 39:list(range(104,140+1)), 40:list(range(104,140+1)), 41:list(range(104,140+1)),

            #42-45
            42:list(range(108,140+1)), 43:list(range(108,140+1)), 44:list(range(108,140+1)), 45:list(range(108,140+1)),

            #46-49
            46:list(range(112,140+1)), 47:list(range(112,140+1)), 48:list(range(112,140+1)), 49:list(range(112,140+1)),

            #50-53
            50:list(range(120,140+1)), 51:list(range(120,140+1)), 48:list(range(120,140+1)), 49:list(range(120,140+1)),

            #54-57
            54:list(range(120,140+1)), 55:list(range(120,140+1)), 56:list(range(120,140+1)), 57:list(range(120,140+1)),

            #58-61
            58:list(range(120,140+1)), 59:list(range(120,140+1)), 60:list(range(120,140+1)), 61:list(range(120,140+1)),

            #86-89
            86:list(range(0,40+1)), 87:list(range(0,40+1)), 88:list(range(0,40+1)), 89:list(range(0,40+1)),

            #90-93
            90:list(range(0,40+1)), 91:list(range(0,40+1)), 92:list(range(0,40+1)), 93:list(range(0,40+1)),

            #94-100
            94:list(range(0,40+1)), 95:list(range(0,40+1)), 96:list(range(0,40+1)), 97:list(range(0,40+1)), 98:list(range(0,40+1)), 99:list(range(0,40+1)), 100:list(range(0,40+1))

        }

    def __init__(self, fp, lookback, iter_chunk_size
                    ,all_at_once=False, start_idx=0, end_idx=None
                    ):
        """Extendable Class handling the generation of model field and rain data
            from E-Obs and ERA5 datasets

        Args:
            fp (str): Filepath of netCDF4 file containing data.
            all_at_once (bool, optional): Whether or not to load all the data in RAM or not. Defaults to False.
            start_idx (int, optional): Skip the first start_idx elements of the dataset
            
        """ 

        if iter_chunk_size%lookback!=0:
            print("Iter chunk size must be a multiple of lookback to ensure that the samples we pass to the LSTM can be transformed to the correct shape")
            iter_chunk_size = lookback* int( ( iter_chunk_size+ + lookback/2)//lookback )

        self.generator = None
        self.all_at_once = all_at_once
        self.fp = fp
        
        #The longitude lattitude grid for the 0.1 degree E-obs and rainfall data
        self.latitude_array = np.linspace(58.95, 49.05, 100)
        self.longitude_array = np.linspace(-10.95, 2.95, 140)
        
        # Retrieving information on temporal length of  dataset        
        with nDataset(self.fp, "r+", format="NETCDF4") as ds:
            if 'time' in ds.dimensions:
                self.max_data_len = ds.dimensions['time'].size
            else:
                raise NotImplementedError
        
        self.lookback = lookback
        self.start_idx = start_idx
        self.end_idx = end_idx if end_idx else self.start_idx + self.max_data_len
        
        # Ensuring we end_idx is a multiple of lookback away
        self.end_idx = self.start_idx+ int(((self.end_idx-self.start_idx)//self.lookback)*self.lookback )

        self.data_len_per_location = self.end_idx - self.start_idx

        self.iter_chunk_size = iter_chunk_size
            
    def yield_all(self):
        pass

    def yield_iter(self):
        pass
    
    def __call__(self, ):
        if(self.all_at_once):
            return self.yield_all()
        else:
            return self.yield_iter()
    
    @lru_cache()
    def find_idxs_of_loc(self, loc="London"):
        """Returns the grid indexes on the 2D map of the UK which correspond to the location (loc) point

        Args:
            loc (str, optional): name of the location. Defaults to "London".

        Returns:
            tuple: Contains indexes (h1,w1) for the location (loc)
        """        
        coordinates = self.city_latlon[loc]
        indexes = self.find_nearest_latitude_longitude( coordinates)  # (1,1)
        return indexes

    def find_idx_of_loc_region(self, loc, dconfig):
        """ Returns the the indexes defining gridded box that surrounds the location of interests

            Raises:
                ValueError: [If the location of interest is too close to the border for evaluation]

            Returns:
                tuple: Returns a tuple ( [upper_h, lower_h], [left_w, right_w] ), defining the grid box that 
                    surrounds the location (loc)
        """
        
        city_idxs = self.find_idxs_of_loc(loc) #[h,w]
        
        # Checking that central region of interest is not too close to the border

        bool_regioncheck_lat = city_idxs[0] > (dconfig.outer_box_dims[0]-dconfig.inner_box_dims[0]) and city_idxs[0] < (city_idxs[0] - (dconfig.outer_box_dims[0]-dconfig.inner_box_dims[0]))
        bool_regioncheck_lon = city_idxs[1] > (dconfig.outer_box_dims[1]-dconfig.inner_box_dims[1]) and city_idxs[1] < (city_idxs[1] - (dconfig.outer_box_dims[1]-dconfig.inner_box_dims[1]))

        # if bool_regioncheck1.any() or bool_regioncheck2.any(): raise ValueError("The specified region is too close to the border")
        if bool_regioncheck_lat or bool_regioncheck_lon: raise ValueError("The specified region is too close to the border")


        # Defining the span, in all directions, from the central region
        if( dconfig.outer_box_dims[0]%2 == 0 ):
            h_up_span = dconfig.outer_box_dims[0]//2 
            h_down_span = h_up_span
        else:
            h_up_span = dconfig.outer_box_dims[0]//2
            h_down_span = dconfig.outer_box_dims[0]//2 + 1

        if( dconfig.outer_box_dims[1]%2 == 0 ):
            w_left_span = dconfig.outer_box_dims[1]//2 
            w_right_span = w_left_span
        else:
            w_left_span = dconfig.outer_box_dims[1]//2
            w_right_span = dconfig.outer_box_dims[1]//2 + 1
        
        #Defining outer_boundaries
        upper_h = city_idxs[0] - h_up_span
        lower_h = city_idxs[0] + h_down_span

        left_w = city_idxs[1] - w_left_span
        right_w = city_idxs[1] + w_right_span
        
        return ( [upper_h, lower_h], [left_w, right_w] )

    
    def find_nearest_latitude_longitude(self, lat_lon):
        """Given specific lat_lon, this method finds the closest long/lat points on the
            0.1degree grid our input/target data is defined on

        Args:
            lat_lon (tuple): tuple containing the lat and lon values of interest

        Returns:
            tuple: tuple containing the idx_h and idx_w values that detail the posiiton on lat_lon on the 
                0.1degree grid on which the ERA5 and E-Obvs data is defined
        """        
        latitude_index =    np.abs(self.latitude_array - lat_lon[0] ).argmin()
        longitude_index =   np.abs(self.longitude_array - lat_lon[1]).argmin()

        return (latitude_index, longitude_index)
        
    def get_locs_for_whole_map(dconfig):
        """This function returns a list of boundaries which can be used to extract all patches
            from the 2D map. 

            Args:
                region_grid_params (dictionary): a dictioary containing information on the sizes of 
                    patches to be extract from the main image

            Returns:
                list:return a list of of tuples defining the boundaries of the region
                        of the form [ ([upper_h, lower_h]. [left_w, right_w]), ... ]
            If we do have a stride then shuffle changes which set of points we pick at each round of iteration
        """      

        original_uk_dim = dconfig.original_uk_dim
        h_shift = dconfig.vertical_shift
        w_shift = dconfig.horizontal_shift
        h_span, w_span = dconfig.outer_box_dims

        #list of values for upper_h and lower_h
        h_start_idx = 0 #if not shuffle else random.randint(0, h_shift-1) //This can not be shuffled since you also have to change the idx_loc_in_region
        range_h = np.arange(h_start_idx, original_uk_dim[0]-h_span, step=h_shift, dtype=np.int32 ) 
        # list of pairs of values (upper_h, lower_h)
        li_range_h_pairs = [ [range_h[i], range_h[i]+h_span] for i in range(0,len(range_h))]
        
        #list of values for left_w and right_w
        w_start_idx = 0 #if not shuffle else random.randint(0, w_shift-1)
        range_w = np.arange(w_start_idx, original_uk_dim[1]-w_span, step=w_shift, dtype=np.int32)
        # list of pairs of values (left_w, right_w)
        li_range_w_pairs = [ [range_w[i], range_w[i]+w_span ] for i in range(0,len(range_w))]

        li_boundaries = list( it.product( li_range_h_pairs, li_range_w_pairs ) ) #[ ([h1,h2],[w1,w2]), ... ]

        filtered_boundaries = Generator.get_filtered_boundaries( li_boundaries )

        return filtered_boundaries
    
    def get_locs_latlon_for_whole_map(self, dconfig):

        filtered_boundaries = Generator.get_locs_for_whole_map(dconfig)

        filtered_latlon = [
            ( 
                [
                    self.hidx_to_lat(hpair_wpair[0][0]),
                    self.hidx_to_lat(hpair_wpair[0][1])
                ],
                [
                    self.widx_to_lon(hpair_wpair[1][0]),
                    self.widx_to_lon(hpair_wpair[1][1])
                ]
            )
            for hpair_wpair in filtered_boundaries
        ]

        return filtered_latlon
    
    def hidx_to_lat(self, hidx):
        return self.latitude_array[hidx]

    def widx_to_lon(self, widx):
        return self.longitude_array[widx]

        


    
    def get_filtered_boundaries(li_boundaries ):

        filtered_boundaries = []
        
        # We filter away obvious training points which are over water
        for h_span, w_span in li_boundaries:
            
            # Find the list of points for ignored points based on the h_posiiton
            relevant_hidxs = [hidx for hidx in Generator.invalid_points.keys() if (hidx>=h_span[0] and hidx<=h_span[1]) ]

            if len(relevant_hidxs)==0: 
                continue
            relevant_wspans = sum( [Generator.invalid_points[hidx] for hidx in relevant_hidxs], [] )

            relevant_wspans = list(set(relevant_wspans))

            w_span_range = list(range(w_span[0], w_span[1]+1))

            # We check if all idxs in w_span are valid, if so we add to filtered_boundaries
            if  all( (widx not in relevant_wspans) for widx in w_span_range ):
                filtered_boundaries.append( (h_span,w_span) )
            else:
                pass

        return filtered_boundaries

class Generator_rain(Generator):
    """ A generator for E-obs 0.1 degree rain data
    
    Returns:
        A python generator for the rain data
        
    """
    def __init__(self, **generator_params ):
        super(Generator_rain, self).__init__(**generator_params)
        
    def yield_all(self):
        """ Return all data at once
        """
        raise NotImplementedError
        with Dataset(self.fp, "r", format="NETCDF4",keepweakref=True) as ds:
            _data = ds.variables['rr'][:]
            yield np.ma.getdata(_data), np.ma.getmask(_data)   
            
    def yield_iter(self):
        """ Return data in chunks"""

        # xr_gn = xr.open_dataset( self.fp, cache=False, decode_times=False, decode_cf=False)
        with xr.open_dataset( self.fp, cache=False, decode_times=False, decode_cf=False, ) as xr_gn:
            idx = copy.deepcopy(self.start_idx)
            # final_idx =  min( self.start_idx+self.data_len_per_location, self.end_idx) 
            # # Same affect as drop_last = True. Ensures that we extract segments with size a mulitple of lookback
            # final_idx = int( self.start_idx + ((final_idx)//self.lookback)*self.lookback )
            # self.data_len_per_location = (final_idx- self.start_idx)
            
            #TODO: Since we are doing strides of lookback length,
            # then add a shuffle ability where we adjust the start idx by up to lookback length so the idx is moved back by at most lookback
            while idx < self.end_idx:

                adj_iter_chunk_size = min(self.iter_chunk_size, (self.end_idx-idx) )
            
                slice_t = slice( idx , idx+adj_iter_chunk_size )
                slice_h = slice( None , None, -1 )
                slice_w = slice( None, None )

                marray = xr_gn.isel(time=slice_t ,latitude=slice_h, longitude=slice_w)['rr'].to_masked_array(copy=True)
                array, mask = np.ma.getdata(marray), np.ma.getmask(marray)

                mask = (array==9.969209968386869e+36)
                
                idx+=adj_iter_chunk_size

                yield array, mask

    def __call__(self):
        return self.yield_iter()
    
    __iter__ = yield_iter
    
class Generator_mf(Generator):
    """Creates a generator for the model_fields_dataset
    """

    def __init__(self, vars_for_feature, **generator_params):
        """[summary]

        Args:
            generator_params:list of params to pass to base Generator class
        """        
        super(Generator_mf, self).__init__(**generator_params)

        self.vars_for_feature = vars_for_feature #['unknown_local_param_137_128', 'unknown_local_param_133_128', 'air_temperature', 'geopotential', 'x_wind', 'y_wind' ]       
        # self.start_idx = 0
        # self.end_idx =0 
        #self.ds = Dataset(self.fp, "r", format="NETCDF4")

    def yield_all(self):
        
        xr_gn = xr.open_dataset(self.fp, cache=False, decode_times=False, decode_cf=False)

        slice_t = slice( self.start_idx , self.end_idx )
        slice_h = slice(1,103-2 )
        slice_w = slice(2,144-2)
        
        xr_gn =  xr_gn.isel(time=slice_t, latitude=slice_h, longitude=slice_w)
        
        return xr_gn

    def yield_iter(self):
        # xr_gn = xr.open_dataset(self.fp, cache=False, decode_times=False, decode_cf=False, cache=False)
        with xr.open_dataset(self.fp, cache=False, decode_times=False, decode_cf=False) as xr_gn:
            idx = copy.deepcopy(self.start_idx)

            while idx < self.end_idx:

                adj_iter_chunk_size = min(self.iter_chunk_size, self.end_idx-idx )

                _slice = slice( idx , idx  + adj_iter_chunk_size)
                next_marray = [ xr_gn[name].isel(time=_slice).to_masked_array(copy=True) for name in self.vars_for_feature ]
                
                list_datamask = [(np.ma.getdata(_mar), np.ma.getmask(_mar)) for _mar in next_marray]
                
                _data, _masks = list(zip(*list_datamask))
                # _masks = [ np.logical_not(_mask_val) for _mask_val in _masks] 
                stacked_data = np.stack(_data, axis=-1)
                stacked_masks = np.stack(_masks, axis=-1)

                idx += adj_iter_chunk_size
                
                yield stacked_data[ :, 1:-2, 2:-2, :], stacked_masks[ :, 1:-2 , 2:-2, :] #(100,140,6) 

    __iter__ = yield_iter 
class Era5EobsDataset(IterableDataset):

    def __init__(self, dconfig, start_date, end_date, target_range,
        locations, loc_count, scaler_features=None, 
        scaler_target=None, workers=1, shuffle=False, **kwargs ) -> None:
        super(Era5EobsDataset).__init__()

        self.dconfig = dconfig
        self.target_range = target_range
        self.locations = locations if locations else dconfig.locations
        self.loc_count = loc_count if loc_count else dconfig.loc_count
        self.end_date = end_date
        start_idx_feat, start_idx_tar = self.get_idx(start_date)
        end_idx_feat, end_idx_tar = self.get_idx(end_date)
        self.gen_size = kwargs.get('gen_size',None)
        self.workers = workers
        self.shuffle = shuffle
        
        # region Checking for pre-existing scalers - If not exists we create in next section
        try:
            trainer_dir = kwargs.get('trainer_dir')
            self.scaler_features = scaler_features if scaler_features != None else pickle.load( open( os.path.join(trainer_dir,"scaler_features.pkl"),"rb") ) 
            self.scaler_target = scaler_target if scaler_target != None else pickle.load( open( os.path.join(trainer_dir,"scaler_target.pkl"),"rb") ) 
        except (FileExistsError, FileNotFoundError, Exception) as e:
            self.scaler_features = None
            self.scaler_target = None
        # endregion
        
        # region Checking for cache of dataset 
        premade_dset_path = os.path.join('Data','uk_rain','premade_dset_record.txt')
        if os.path.exists(premade_dset_path):
            premade_dsets = pd.read_csv( premade_dset_path)
        else:
            premade_dsets = pd.DataFrame( columns=['cache_path','scaler_path','start_date','end_date','locations','lookback',
                                                    'target_range','outer_box_dims'] )

        # Query for if cache for dataset and normalizer
        query_res = premade_dsets.query( f"start_date == '{start_date}' and end_date == '{end_date}' and \
                                    locations == '{'_'.join([loc[:2] for loc in self.locations])}' and \
                                    lookback == {str(self.dconfig.lookback_target)} and \
                                    target_range == '{','.join(map(str,target_range))}' and \
                                    outer_box_dims == '{','.join(map(str,dconfig.outer_box_dims))}'")
        
        #If it exists we just load it
        if len(query_res)!=0:
            
            self.cache_path = query_res['cache_path'].iloc[0]
            self.cache_exists = os.path.exists(self.cache_path)
        
            # Otherwise set up parameters to use the normal dataset
        else:
            os.makedirs( os.path.join(dconfig.data_dir, "cache"), exist_ok=True )
            self.cache_path = os.path.join( self.dconfig.data_dir, "cache",
                f"start-{start_date}-end-{end_date}_lbtarget-{str(self.dconfig.lookback_target)}-tgtrange-{','.join(map(str,target_range))}-outer_box_dims-{','.join(map(str,dconfig.outer_box_dims))} -locs-{'_'.join([loc[:2] for loc in self.locations])}")
            self.cache_exists = os.path.exists(self.cache_path)
        # endregion

        #Finally if any of cache or scalers don't exists we create the non existent ones
        if not self.cache_exists or self.scaler_features == None or self.scaler_target ==  None:
            # Create python generator for rain data
            fp_rain = self.dconfig.data_dir+"/"+self.dconfig.rain_fn
            self.rain_data = Generator_rain(fp=fp_rain,
                                            all_at_once=False,
                                            iter_chunk_size=self.dconfig.lookback_target*self.gen_size,
                                            lookback=self.dconfig.lookback_target,
                                            start_idx=start_idx_tar,
                                            end_idx = end_idx_tar
                                            )

            # Create python generator for model field data 
            mf_fp = self.dconfig.data_dir + "/" + self.dconfig.mf_fn
            self.mf_data = Generator_mf(fp=mf_fp, vars_for_feature=self.dconfig.vars_for_feature, 
                                        all_at_once=False, 
                                        iter_chunk_size=self.dconfig.lookback_feature*self.gen_size,
                                        lookback=self.dconfig.lookback_feature,
                                        start_idx=start_idx_feat,
                                        end_idx = end_idx_feat)
            
            self.create_cache_scaler()
        
        self.create_cache_params()
            
        if not self.cache_exists or len(query_res)==0:
            premade_dsets = premade_dsets.append( {
                'cache_path':self.cache_path,
                'start_date':start_date,
                'end_date':end_date,
                'locations':'_'.join([loc[:2] for loc in self.locations]),
                'lookback':str(self.dconfig.lookback_target),
                'target_range':','.join(map(str,target_range)),
                'outer_box_dims':','.join(map(str,dconfig.outer_box_dims) )
            }, ignore_index=True)
            
            premade_dsets.to_csv( premade_dset_path, index=False)
                   
        # Create buffers for scaler params
        self.features_scale = torch.as_tensor( self.scaler_features.scale_ )
        self.features_mean = torch.as_tensor( self.scaler_features.mean_ )
        self.target_scale = torch.as_tensor( self.scaler_target.scale_ )

    @staticmethod
    def get_dataset( dconfig, target_range, target_distribution_name,**kwargs):

        ds_train = Era5EobsDataset( start_date = dconfig.train_start, end_date=dconfig.train_end,
                                    target_distribution_name=target_distribution_name,
                                    locations=dconfig.locations,
                                    loc_count=dconfig.loc_count,
                                    target_range=target_range, dconfig=dconfig, 
                                    shuffle=dconfig.shuffle,**kwargs )
        
        ds_val = Era5EobsDataset( start_date = dconfig.val_start, end_date=dconfig.val_end,
                                    dconfig=dconfig,
                                    target_range=target_range,
                                    target_distribution_name=target_distribution_name,
                                    locations=dconfig.locations,
                                    loc_count=dconfig.loc_count,
                                    scaler_features = ds_train.scaler_features,
                                    scaler_target = ds_train.scaler_target,
                                    shuffle=False,
                                    **kwargs)

        # assert dconfig.locations_test != ["All"], "Can not test over Whole map. please consider using `All_Cities"
        ds_test = Era5EobsDataset( start_date=dconfig.test_start, end_date=dconfig.test_end,
                                    locations=dconfig.locations_test, loc_count=dconfig.loc_count,
                                    dconfig=dconfig,
                                    target_range=target_range,
                                    target_distribution_name=target_distribution_name,
                                    scaler_features = ds_train.scaler_features,
                                    scaler_target = ds_train.scaler_target,
                                    shuffle=False,
                                    **kwargs)

        return ds_train, ds_val, ds_test, ds_train.scaler_features, ds_train.scaler_target
    
    @staticmethod
    def get_test_dataset( dconfig, target_range, target_distribution_name, scaler_features, scaler_target, **kwargs):

        # assert dconfig.locations_test != ["All"], "Can not test over Whole map. please consider using `All_Cities"
        ds_test = Era5EobsDataset( start_date=dconfig.test_start, end_date=dconfig.test_end,
                                    locations=dconfig.locations_test, loc_count=dconfig.loc_count,
                                    dconfig=dconfig,
                                    target_range=target_range,
                                    target_distribution_name=target_distribution_name,
                                    scaler_features = scaler_features,
                                    scaler_target = scaler_target,
                                    shuffle=False,
                                    **kwargs)

        return  ds_test

    def __iter__(self):

        if self.cache_exists and self.scaler_features and self.scaler_target:
          
            with xr.open_dataset( self.cache_path, cache=True, ) as xr_cache:
                # for feature, target, target_mask, idx_loc_in_region in self.cached_data:
                
                if not hasattr(self, 'max_cache_len'):
                    # for feature, target, target_mask, idx_loc_in_region in self.cached_data:
                    self.max_cache_len = xr_cache["sample_idx"].shape[0]

                # Making sure its a multiple of lookback away from start idx
                if not hasattr(self, 'cache_start_idx'):
                    self.cache_start_idx = 0

                if not hasattr(self, 'cache_end_idx'):
                    self.cache_end_idx = 0 + int(((self.max_cache_len - self.cache_start_idx)//self.lookback)*self.lookback )

                if not hasattr(self, 'cache_len'):
                    self.cache_len = self.cache_end_idx + 1 - self.cache_start_idx
                
                # # Adding Shuffling
                # # This is shuffling in sequence of days we pass to the model
                if self.shuffle:
                    # We adjust the start idx of model field and rain data
                    # The amount to increment the target, e.g. number of days 
                    # Debug
                    self.target_sub_idx_increment = random.randint(0, self.dconfig.lookback_target)
                    self.feature_sub_idx_increment = self.target_sub_idx_increment*4
          
                idx = copy.deepcopy(self.cache_start_idx)

                while idx < self.cache_end_idx :
                    adj_iter_chunk_size = min( self.gen_size, self.cache_end_idx-idx )

                    _slice = slice( idx , idx  + adj_iter_chunk_size)
                    dict_data = { name:torch.tensor( xr_cache[name].isel(sample_idx=_slice).data )
                                    for name in ['input','target','mask','idx_loc_in_region'] }

                    dict_data['li_locations'] = xr_cache['li_locations'].isel(sample_idx=_slice).data 
                                                      
                    dict_data['input'] = (dict_data['input'] - self.features_mean )/self.features_scale
                    dict_data['target'] = dict_data['target']*self.target_scale
                    
                    #scaling
                    dict_data['input'] = dict_data['input'].to(torch.float16).squeeze()
                    dict_data['target'] = dict_data['target'].to(torch.float16).squeeze()
                    dict_data['mask'] = dict_data['mask'].to(torch.bool).squeeze()

                    # dict_data = { k:v for k,v in dict_data.items() }
                    idx += adj_iter_chunk_size
                    
                    # converting from dictionary of keys, where each key contains info for all timesteps to
                    # list of dicts where each dict contains all keys and info for one timestep
                    if type(dict_data['input'])==tuple:
                        li_dicts = [ {key:dict_data[key][idx].unsqueeze(0) if key!="li_locations" else dict_data[key][idx] for key in dict_data.keys()} for idx in range(len(dict_data['target'])) ]
                      
                    elif adj_iter_chunk_size==1:
                        dict_data['idx_loc_in_region'] = dict_data['idx_loc_in_region'].squeeze(0)
                        li_dicts = [ {k:v.unsqueeze(0) if k!="li_locations" else v for k,v in dict_data.items()} ]
                        
                    else:
                        _ =  {k:v.unbind() if k!="li_locations" else v for k,v in dict_data.items()} 
                        li_dicts = [ {key:_[key][idx].unsqueeze(0) if key!="li_locations" else _[key][idx] for key in dict_data.keys() } for idx in range(len(dict_data['target'])) ]
                    
                    # shuffling data 
                    # unbundling weekly batches and shifting data forward n days 
                    # such that the weekly periods are different
                    if self.shuffle and self.target_sub_idx_increment>0:
                        li_dicts = self.cache_shuffle(li_dicts)

                    # Filter li_dicts that do not have any valid valuesju
                    li_dicts = [ dict_ for dict_ in li_dicts if dict_['mask'].logical_not().any()]
                    
                    yield from li_dicts
                        
        else: 
            print("Note: User is using non cached dataset. The data will not be normalized automatically")
            sample_idx = 0
            # We calculate the maximum number of samples as the start_idx - end_idx // self.lookback for each generator
            # sample_max_len = min( [ (gen.data_len_per_location)//gen.lookback for gen in [self.mf_data, self.rain_data]] )

            if self.scaler_features == None: 
                self.scaler_features = StandardScaler()
                bool_update_scaler_features = True
            else:
                bool_update_scaler_features = False

            if self.scaler_target == None: 
                self.scaler_target = MinMaxScaler(feature_range=self.target_range)
                bool_update_scaler_target = True
            else:
                bool_update_scaler_target = False
            
            for idx, ( (feature, feature_mask), (target, target_mask) ) in enumerate( zip( self.mf_data, self.rain_data)):

                dict_data = self.preprocess_batch(feature, feature_mask, target, target_mask, normalize=False)

                #Developing Data Cache and Scalers
                if not self.cache_exists:
                    kwargs ={ 
                        "data_vars":{
                                "input":( ("sample_idx","lookback_feature","h_feat","w_feat","d"), torch.concat(dict_data['input']).numpy() ),
                                "target": ( ("sample_idx","lookback_target","h_target","w_target"),torch.concat(dict_data['target']).numpy() ),
                                "mask": ( ("sample_idx","lookback_target","h_target","w_target"),torch.concat(dict_data['mask']).numpy() ),
                                "idx_loc_in_region": ( ("sample_idx","h_w"), np.concatenate(dict_data['idx_loc_in_region']) ) ,
                                "li_locations": ( ("sample_idx"), dict_data['li_locations'] )
                                }
                            }
                                           
                    if idx==0 or os.path.exists(self.cache_path):
                        
                        coords = {
                            "sample_idx": np.arange( torch.concat(dict_data['input']).shape[0] ),
                            "lookback_feat": np.arange( self.dconfig.lookback_feature),
                            "lookback_target": np.arange( self.dconfig.lookback_target),
                            "h_feat": np.arange( dict_data['input'][0].shape[-3]),
                            "w_feat": np.arange( dict_data['input'][0].shape[-2]),
                            "d": np.arange( dict_data['input'][0].shape[-1]),
                            "h_target": np.arange( dict_data['target'][0].shape[-2]),
                            "w_target": np.arange( dict_data['target'][0].shape[-1]),
                            "h_w": np.arange( dict_data['idx_loc_in_region'][0].shape[-1] )
                            }
                        kwargs['coords'] = coords
                        xr_curr = xr.Dataset( **kwargs )

                    else:

                        kwargs['coords'] = {
                            "sample_idx": np.arange( xr_curr.dims['sample_idx'], xr_curr.dims['sample_idx']+torch.concat(dict_data['input']).shape[0]  ),

                            "lookback_feat": np.arange( self.dconfig.lookback_feature),
                            "lookback_target": np.arange( self.dconfig.lookback_target),

                            "h_feat": np.arange( dict_data['input'][0].shape[-3]),
                            "w_feat": np.arange( dict_data['input'][0].shape[-2]),

                            "h_target": np.arange( dict_data['target'][0].shape[-2]),
                            "w_target": np.arange( dict_data['target'][0].shape[-1]),
                            "h_w": np.arange( dict_data['idx_loc_in_region'][0].shape[-1] ),

                            "d": np.arange( dict_data['input'][0].shape[-1]),
                        }

                        xr_new = xr.Dataset( **kwargs)
                        xr_curr = xr.concat( [ xr_curr, xr_new], dim="sample_idx", join="exact" )

                if bool_update_scaler_features: 
                    # reshaping feature into ( num, dims) dimension required by partial_fit
                    dim = dict_data['input'][0].shape[-1]
                    features = torch.concat(dict_data['input'])
                    features_numpy = features.numpy()
                    # make sure to only take inner dim if applicable and to use feature_mask
                    if features_numpy.ndim == 5:
                        bounds = Era5EobsDataset.central_region_bounds(self.dconfig) #list [ lower_h_bound[0], upper_h_bound[0], lower_w_bound[1], upper_w_bound[1] 
                        features_numpy = torch.stack( [
                            Era5EobsDataset.extract_central_region(feature, bounds )
                                for feature in torch.unbind(features, dim=-1 )
                                ]
                            ,dim=-1)

                    self.scaler_features.partial_fit( features_numpy.reshape(-1, dim ) )
                
                if bool_update_scaler_target:
                    dim = 1    
                    target = torch.concat(dict_data['target'])
                    target_mask = torch.concat(dict_data['mask'])

                    target = target[~target_mask]
                    if target.numel()!=0:
                        self.scaler_target.partial_fit( target.reshape(-1, dim).numpy() )

                sample_idx += torch.concat(dict_data['input']).shape[0]
                
                # yield dict_data
                if type(dict_data['input'])==tuple:
                    li_dicts = [ {key:dict_data[key][idx] for key in dict_data.keys() }for idx in range(len(dict_data['target'])) ]
                    yield from li_dicts
                
                else:
                    yield dict_data

            if not self.cache_exists:
                comp = dict(zlib=True, complevel=5)
                encoding = {var: comp for var in xr_curr.data_vars}
                xr_curr.to_netcdf(self.cache_path, mode='w', encoding=encoding )
                
    def cache_shuffle(self, li_dicts):
        # shuffling data 
        # unbundling weekly batches and shifting data forward n days 
        # such that the weekly periods are different
        keys = list(li_dicts[0].keys())

        dict_loc_unbundled = {}
        dict_loc_batched = {}

        # Unbundling each variable for each location
        # Converting it into one long sequence for each variable for a location instead of chunks of 7 day data
        for loc in self.locations:
            loc_dict = { }

            location_data = [ dict_ for dict_ in li_dicts if dict_["li_locations"]==loc ]
            if len(location_data) ==0: 
                continue

            for key in keys:
                
                #Concat all elements for each variable
                if isinstance( li_dicts[0][key], torch.Tensor ):
                    
                    if key in ["input"]:
                        key_data = torch.cat( [_dict[key].squeeze(0) for _dict in location_data], dim=0 ) 

                    elif key in ["idx_loc_in_region"]:
                        key_data = torch.stack( [_dict[key].squeeze(0) for _dict in location_data], dim=0 )
                    
                    elif key in ['target']:
                        key_data = torch.cat( [_dict[key].squeeze(0) for _dict in location_data], dim=0 ) 
                    
                    elif key in ['mask']:
                        key_data = torch.cat( [_dict[key].squeeze(0) for _dict in location_data], dim=0 )

                else:
                    key_data = [_dict[key] for _dict in location_data]
                                
                loc_dict[key] = key_data
                    
            dict_loc_unbundled[loc] = loc_dict
        
        # rebatching into weeks with a n day increment
        for loc in dict_loc_unbundled.keys():
            dict_loc_batched[loc] = {}
            # incrementing and batching
            for key in keys:
                
                if key in ['target','mask']:
                    dict_loc_unbundled[loc][key] = dict_loc_unbundled[loc][key][self.target_sub_idx_increment:]
                    d = dict_loc_unbundled[loc][key]
                    l = len(d)-self.dconfig.lookback_target
                    dict_loc_batched[loc][key] = [ d[idx:idx+self.dconfig.lookback_target].unsqueeze(0) for idx in range(0, l, self.dconfig.lookback_target) ]

                elif key in ['idx_loc_in_region']:
                    # idx_loc_in_region only has one value per lookback_target.
                    s_dix = round( self.target_sub_idx_increment/self.dconfig.lookback_target )
                    d = dict_loc_unbundled[loc][key][s_dix:]
                    l = len(d)
                    dict_loc_batched[loc][key] = [ d[idx:idx+1] for idx in range(0, l, 1) ]

                elif key in ['li_locations']:
                    s_dix = round( self.target_sub_idx_increment/self.dconfig.lookback_target )
                    d = dict_loc_unbundled[loc][key][s_dix:]
                    l = len(d)
                    dict_loc_batched[loc][key] = [ d[idx:idx+1] for idx in range(0,l,1) ]

                elif key in ['input']:
                    dict_loc_unbundled[loc][key] = dict_loc_unbundled[loc][key][self.feature_sub_idx_increment:]
                    d = dict_loc_unbundled[loc][key]
                    l = len(d)-self.dconfig.lookback_feature
                    dict_loc_batched[loc][key] = [ d[idx:idx+self.dconfig.lookback_feature].unsqueeze(0) for idx in range(0, l, self.dconfig.lookback_feature) ]

        count = sum( len( dict_loc_batched[loc][keys[0]] ) for loc in dict_loc_batched.keys() )
        li_dicts_shuffled = [ {} for i in range(count) ]
        
        # Creating a list of dicts structure - where each dict contains data for one 7 day period
        for key in keys:
            
            if key in ['target','mask']:
                li_datums = sum( [ dict_loc_batched[loc][key] for loc in dict_loc_batched.keys() ], [] )
            elif key in ['input']:
                li_datums = sum( [ dict_loc_batched[loc][key] for loc in dict_loc_batched.keys() ], [] )
            elif key in ['idx_loc_in_region']:
                li_datums = sum( [ dict_loc_batched[loc][key] for loc in dict_loc_batched.keys() ], [] )
            elif key == ['li_locations']:
                li_datums = sum( [ dict_loc_batched[loc][key] for loc in dict_loc_batched.keys() ], [] )

            for idx in range(count):
                li_dicts_shuffled[idx][key] = li_datums[idx]
        
        return li_dicts_shuffled

    def preprocess_batch(self, feature, feature_mask, target, target_mask, normalize=True):
        
        # Converting to tensors
        feature = torch.as_tensor(feature)
        feature_mask = torch.as_tensor(feature_mask)
        target = torch.as_tensor(target)
        target_mask = torch.as_tensor(target_mask)

        # Preparing feature model fields
        # unbatch
        feature = feature.view(-1, self.dconfig.lookback_feature ,*feature.shape[-3:] ) # ( bs*feat_days ,h, w, shape_feat)
        feature_mask = feature_mask.view(-1, self.dconfig.lookback_feature,*feature.shape[-3:] ) # ( bs*feat_days ,h, w, shape_dim)
        
        if normalize:
            feature = (feature-self.feature_mean )/self.featurses_scale
        feature.masked_fill_( feature_mask, self.dconfig.mask_fill_value['model_field'])

        # Preparing Eobs and target_rain_data
        target = target.view(-1, self.dconfig.lookback_target, *target.shape[-2:] ) #( bs, target_periods ,h1, w1, target_dim)
        target_mask = target_mask.view(-1, self.dconfig.lookback_target, *target.shape[-2:] )#( bs*target_periods ,h1, w1, target_dim)

        if normalize:
            target = target*self.target_scale
        target.masked_fill_(target_mask, self.dconfig.mask_fill_value['rain'] )

        li_feature, li_target, li_target_mask, idx_loc_in_region, li_locs = self.location_extractor( feature, target, target_mask, self.locations)
        
        dict_data = { k:v for k,v in zip(['input','target','mask','idx_loc_in_region','li_locations' ], [li_feature, li_target, li_target_mask, idx_loc_in_region, li_locs] ) }
        return dict_data

    def get_idx(self, date:Union[np.datetime64,str]):
        """ Returns two indexes
                The first index is the idx at which to start extracting data from the feature dataset
                The second index is the idx at which to start extracting data from the target dataset
            Args:
                start_date (np.datetime64): Start date for evaluation
            Returns:
                tuple (int, int): starting index for the feature, starting index for the target data
        """        

        if type(date)==str:
            date = copy.deepcopy(date)
            date = np.datetime64(date)
            
        feature_start_date = self.dconfig.feature_start_date
        target_start_date = self.dconfig.target_start_date

        feat_days_diff = np.timedelta64(date - feature_start_date,'6h').astype(int)
        tar_days_diff = np.timedelta64(date - target_start_date, 'D').astype(int)

        feat_idx = feat_days_diff #since the feature comes in four hour chunks
        tar_idx = tar_days_diff 

        return feat_idx, tar_idx

    def location_extractor(self, feature, target, target_mask, locations):
        """Extracts the temporal slice of patches corresponding to the locations of interest 

                Args:
                    ds (tf.Data.dataset): dataset containing temporal slices of the regions surrounding the locations of interest
                    locations (list): list of locations (strings) to extract

                Returns:
                    tuple: (tf.data.Dataset, [int, int] ) tuple containing dataset and [h,w] of indexes of the central region
        """        
                
        # list of central h,w indexes from which to extract the region around
        if locations == ["All"]:
            li_hw_idxs = Generator.get_locs_for_whole_map( self.dconfig ) #[ ([upper_h, lower_h]. [left_w, right_w]), ... ]
            
            locs_latlon_for_whole_map = self.rain_data.get_locs_latlon_for_whole_map(self.dconfig) #[ ([lat1, lat2]. [lon1, lon2]), ... ]

            #Convert to string format
            locs_latlon_for_whole_map = [  
                'lat_' + '_'.join(map("{:.2f}".format,lilat_lilon[0])) + '_lon_' + '_'.join(map("{:.2f}".format,lilat_lilon[1]))
                for lilat_lilon in locs_latlon_for_whole_map ]
            
            li_locs = np.repeat(locs_latlon_for_whole_map, len(target) )

        else:
            li_hw_idxs = [ self.rain_data.find_idx_of_loc_region( _loc, self.dconfig ) for _loc in locations ] #[ (h_idx,w_idx), ... ]
            li_locs = np.repeat(locations, len(target) )

        # Creating seperate datasets for each location
        li_feature, li_target, li_target_mask = zip(*[self.select_region( feature, target, target_mask, hw_idxs[0], hw_idxs[1] ) for hw_idxs in li_hw_idxs ] )

        # pair of indexes locating the central location within the grid region extracted for any location
        idx_loc_in_region = [ np.floor_divide( self.dconfig.outer_box_dims, 2)[np.newaxis,...] ]*len( torch.concat(li_feature) ) #This specifies the index of the central location of interest within the (h,w) patch    
        
        return li_feature, li_target, li_target_mask, idx_loc_in_region, li_locs
    
    def select_region( self, mf, rain, rain_mask, h_idxs, w_idxs):
        """ Extract the region relating to a [h_idxs, w_idxs] pair

            Args:
                mf:model field data
                rain:target rain data
                rain_mask:target rain mask
                h_idxs:int
                w_idxs:int

            Returns:
                tf.data.Dataset: 
        """

        """
            idx_h,idx_w: refer to the top left right index for the square region of interest this includes the region which is removed after cropping to calculate the loss during train step
        """
    
        mf = mf[ ..., h_idxs[0]:h_idxs[1] , w_idxs[0]:w_idxs[1] ,:] # (shape, h, w, d)
        rain = rain[ ..., h_idxs[0]:h_idxs[1] , w_idxs[0]:w_idxs[1] ]
        rain_mask = rain_mask[ ..., h_idxs[0]:h_idxs[1] , w_idxs[0]:w_idxs[1] ]
            
        return mf, rain, rain_mask #Note: expand_dim for unbatch/batch compatibility

    def create_cache_scaler(self):

        """
            This functions returns the cache_path, scaler_features, scaler_target
            if self.scaler_features, or self.scaler_target do not exists then they are created

        Returns:
            [type]: [description]
        """
        
        # if os.path.exists(self.cache_path):
        #     os.remove(self.cache_path)
        # self.cache_exists = False

        for dict_data in iter(self):
            pass

        self.cache_exists = True
        
        assert self.scaler_features 
        assert self.scaler_target
        assert os.path.exists(self.cache_path) 

        return self.scaler_features, self.scaler_target

    def create_cache_params(self):
        
        with xr.open_dataset( self.cache_path, cache=True ) as xr_cache:
            # for feature, target, target_mask, idx_loc_in_region in self.cached_data:
            if not hasattr(self,'max_cache_len'):
                self.max_cache_len = xr_cache["sample_idx"].shape[0]

        # Making sure its a multiple of lookback away from start idx
        if not hasattr(self,'cache_start_idx'):
            self.cache_start_idx = 0
        if not hasattr(self,'cache_end_idx'):
            if self.shuffle:
                self.cache_end_idx = 0 + int(((self.max_cache_len - self.cache_start_idx)//self.dconfig.lookback_target)*self.dconfig.lookback_target )
            else:
                self.cache_end_idx = 0 + self.max_cache_len
        if not hasattr(self,'cache_len'):
            self.cache_len = self.cache_end_idx + 1 - self.cache_start_idx

    @staticmethod
    def worker_init_fn(worker_id:int):
        
        worker_info = torch.utils.data.get_worker_info()
        worker_count = worker_info.num_workers

        if isinstance(worker_info.dataset, ShufflerIterDataPipe):
            # offers backward compatability for pytorch lightning pre v
            ds = worker_info.dataset.datapipe.iterable
        elif isinstance(worker_info.dataset, Era5EobsDataset ):
            ds = worker_info.dataset
        elif isinstance(worker_info.dataset, _IterDataPipeSerializationWrapper ):
            ds = worker_info.dataset._datapipe.datapipe.iterable
        else:
            raise ValueError

        if ds.cache_exists:
            per_worker = ds.cache_len//worker_count
            ds.cache_start_idx = per_worker * worker_id
            ds.cache_end_idx = per_worker * ( worker_id + 1)
            ds.cache_len_per_worker = per_worker
        
        else:
            # Changing the start_idx and end_idx in the underlying generators
            mf_data_len_per_location = ds.mf_data.start_idx - ds.mf_data.end_idx
            per_worker_per_location = mf_data_len_per_location//worker_count 
            ds.mf_data.start_idx = worker_id*per_worker_per_location
            ds.mf_data.end_idx = (worker_id+1)*per_worker_per_location
            ds.mf_data.data_len_per_worker = per_worker_per_location * ds.loc_count

            rain_data_len_per_location = ds.rain_data.start_idx - ds.rain_data.end_idx
            per_worker_per_location = rain_data_len_per_location//worker_count 
            ds.rain_data.start_idx = worker_id*per_worker_per_location
            ds.rain_data.end_idx = (worker_id+1)*per_worker_per_location
            ds.rain_data.data_len_per_worker = per_worker_per_location * ds.loc_count

    @staticmethod
    def parse_data_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=True, allow_abbrev=False)

        parser.add_argument("--original_uk_dim", default=(100,140) )
        parser.add_argument("--input_shape", default=(6,), type=tuple_type ) #TODO: need to roll together input_shape and outer_box_dim logic into one variable. Currently it only refers to the channel depth variable 
        parser.add_argument("--output_shape", default=(1,1), type=tuple_type ) #NOTE: this refers to the depth of the output i.e. do we predict mean and var term or just mean

        parser.add_argument("--locations", nargs='+', required=True)
        parser.add_argument("--locations_test", nargs='+', required=False, default=["London","Cardiff","Glasgow","Manchester","Birmingham","Liverpool","Edinburgh","Dublin","Preston","Truro","Bangor","Plymouth","Norwich","StDavids","Salford","Hull" ] )

        parser.add_argument("--data_dir", default="./Data/uk_rain", type=str)
        parser.add_argument("--rain_fn", default="eobs_true_rainfall_197901-201907_uk.nc", type=str)
        parser.add_argument("--mf_fn", default="model_fields_linearly_interpolated_1979-2019.nc", type=str)

        parser.add_argument("--vertical_shift", type=int, default=1)
        parser.add_argument("--horizontal_shift", type=int, default=1)

        parser.add_argument("--outer_box_dims", default=[1,1], type=tuple_type)
        parser.add_argument("--inner_box_dims", default=[1,1], type=tuple_type)

        parser.add_argument("--lookback_target", type=int, default=7)
        # parser.add_argument("--target_range", nargs='+', default=[0,4])

        parser.add_argument("--train_start", type=str, default="1979")
        parser.add_argument("--train_end", type=str, default="2009")

        parser.add_argument("--val_start", type=str, default="2009")
        parser.add_argument("--val_end", type=str, default="2014")

        parser.add_argument("--test_start", type=str, default="2014")
        parser.add_argument("--test_end", type=str, default="2019-07")

        parser.add_argument("--min_rain_value", type=float, default=0.5)
        parser.add_argument("--gen_size", type=int, default=60, help="Chunk size when slicing the netcdf fies for model fields and rain. When training over many locations, make sure to use large chunk size.")

        parser.add_argument("--shuffle", type=lambda x: bool(int(x)), default=True, choices=[0,1] )
        dconfig = parser.parse_known_args()[0]
        dconfig.locations = sorted(dconfig.locations)
        dconfig.locations_test = sorted(dconfig.locations_test)
        
        #TODO: Add check to ensre that original_shape dims is divisible by  inner_box dims and outer_box dims
        ## iff using two dimensions in input shape or implement method whereby overall shape is cropped to allows for any input_shape dims
        
        dconfig = Era5EobsDataset.add_fixed_args(dconfig)
        
        return dconfig
    
    @staticmethod
    def add_fixed_args(dconfig):
        dconfig.mask_fill_value = {
                                    "rain":0.0,
                                    "model_field":0.0 
        }
        dconfig.vars_for_feature = ['unknown_local_param_137_128', 'unknown_local_param_133_128', 'air_temperature', 'geopotential', 'x_wind', 'y_wind' ]
        

        dconfig.window_shift = dconfig.lookback_target
        
        #Input is at 6 hour intervals, target is daily
        dconfig.lookback_feature = dconfig.lookback_target*4
        
        dconfig.target_start_date = np.datetime64('1950-01-01') + np.timedelta64(10592,'D')
        dconfig.feature_start_date = np.datetime64('1970-01-01') + np.timedelta64(78888, 'h')

        # a string containing four dates seperated by underscores
        # The numbers correspond to trainstart_trainend_valstart_valend

        train_start_date = np.datetime64(dconfig.train_start,'D')
        train_end_date = (pd.Timestamp(dconfig.train_end) - pd.DateOffset(seconds=1) ).to_numpy()

        val_start_date = np.datetime64(dconfig.val_start,'D')
        val_end_date = (pd.Timestamp(dconfig.val_end) - pd.DateOffset(seconds=1) ).to_numpy()

        test_start_date = np.datetime64(dconfig.test_start,'D')
        test_end_date = (pd.Timestamp(dconfig.test_end) - pd.DateOffset(seconds=1) ).to_numpy()

        loc_count = len(dconfig.locations)  if \
                    dconfig.locations != ["All"] else \
                    len( Generator.get_locs_for_whole_map(dconfig))

        loc_count_test = loc_count \
                            if not dconfig.locations_test \
                            else (
                                    len(dconfig.locations_test)  if \
                                        dconfig.locations_test != ["All"] else \
                                        len( Generator.get_locs_for_whole_map(dconfig))
                                    )
        
        dconfig.loc_count = loc_count
        dconfig.loc_count_test = loc_count_test

        dconfig.train_set_size_elements = ( np.timedelta64(train_end_date - train_start_date,'D') // dconfig.window_shift ).astype(int)
        dconfig.train_set_size_elements *= loc_count

        dconfig.val_set_size_elements = ( np.timedelta64(val_end_date - val_start_date,'D')  // dconfig.window_shift  ).astype(int)               
        dconfig.val_set_size_elements *= loc_count

        dconfig.test_set_size_elements = ( np.timedelta64(test_end_date - test_start_date,'D')  // dconfig.window_shift  ).astype(int)               
        dconfig.test_set_size_elements *= loc_count_test
        
        if dconfig.locations[0] == "All_Cities":
            dconfig.locations = sorted( list( Generator.city_latlon.keys() ) )

        if dconfig.locations_test[0] == "All_Cities":
            dconfig.locations_test = sorted( list( Generator.city_latlon.keys() ) )
        

        return dconfig

    @staticmethod
    def cond_rain(vals, probs, threshold=0.5):
        """
            If prob of event occuring is above 0.5 return predicted conditional event value,
            If it is below 0.5, then return 0
        """
        round_probs = torch.where(probs<=threshold, 0.0, 1.0)
        vals = vals* round_probs
        return vals
    
    @staticmethod
    # @lru_cache(100)
    def central_region_bounds(dconfig):
        """Returns the indexes defining the boundaries for the central regions for evaluation

        Args:
            dconfig (dict): information on formualation of the patches used in this ds 

        Returns:
            list: defines the vertices of the patch for extraction
        """    

        central_hw_point = np.asarray(dconfig.outer_box_dims)//2
        
        lower_hw_bound = central_hw_point - np.asarray(dconfig.inner_box_dims) //2

        upper_hw_bound = lower_hw_bound + np.asarray(dconfig.inner_box_dims )
        

        return [lower_hw_bound[0], upper_hw_bound[0], lower_hw_bound[1], upper_hw_bound[1]]

    @staticmethod
    def extract_central_region(tensor, bounds):
        """
            Args:
                tensor ([type]): 4d 
                bounds ([type]): bounds defining the vertices of the patch to be extracted for evaluation
        """
        tensor_cropped = tensor[ ..., bounds[0]:bounds[1],bounds[2]:bounds[3]  ]     #(bs, h , w)
        return tensor_cropped
    
    @staticmethod
    def water_mask( tensor, mask, mask_val=np.nan):
        """Mask out values in tensor by with mask value=0.0
        """
        tensor = torch.where(mask, tensor, mask_val)

        return tensor


# endrefion
MAP_NAME_DSET = {'toy':ToyDataset, 'australia_rain':AustraliaRainDataset, 'uk_rain':Era5EobsDataset }

