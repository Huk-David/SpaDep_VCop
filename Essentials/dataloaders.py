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
import os
import ujson
import pickle
import regex  as re
from typing import Tuple, Callable,  Union, Dict
from torchtyping import TensorDetail,TensorType
import argparse
import json

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
            
            'mv_uniform': lambda lb=0, ub=1 : td.Independent( td.Uniform(lb, ub), 1 ),

            'mv_normal': lambda loc=torch.zeros((6,)), covariance_matrix=torch.eye(6): td.MultivariateNormal( loc , covariance_matrix ),
            
            'mv_lognormal': lambda loc=torch.zeros((6,)), scale=torch.ones( (6,) ) : td.Independent( td.LogNormal( loc , scale ), 1 )

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
            # TODO: Ensure scaler features is only trained on training set
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

MAP_NAME_DSET = {'toy':ToyDataset, 'australia_rain':AustraliaRainDataset}