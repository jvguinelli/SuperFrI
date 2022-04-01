import numpy as np
import random as rd
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import time as tm
import os

from model.stconvs2s import STConvS2S_R, STConvS2S_C
from model.baselines import *
from model.ablation import *
 
from tool.train_evaluate import Trainer, Evaluator
from tool.dataset import CORDataset
from tool.loss import RMSELoss, CustomMSELoss
from tool.utils import Util

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim

import h5py

class MLBuilder:

    def __init__(self, config, device):
        
        self.config = config
        self.device = device
        self.dataset_type = 'small-dataset' if (self.config.small_dataset) else 'full-dataset'
        self.x_step = config.x_step
        self.y_step = config.y_step
        self.dataset_name, self.dataset_file = self.__get_dataset_file()
        self.dropout_rate = self.__get_dropout_rate()
        self.filename_prefix = f"{self.dataset_name}"
                
    def run_model(self, number):
        self.__define_seed(number)
        # Loading the dataset
        #ds = xr.open_mfdataset(self.dataset_file)

        with h5py.File(self.dataset_file, 'r') as f:
            x_treino = f['/x_treino'][...]
            y_treino = f['/y_treino'][...]

            x_validacao = f['/x_validacao'][...]
            y_validacao = f['/y_validacao'][...]

            x_teste = f['/x_teste'][...]
            y_teste = f['/y_teste'][...]

        train_dataset = CORDataset(x_treino, y_treino)
        val_dataset   = CORDataset(x_validacao, y_validacao)
        test_dataset  = CORDataset(x_teste, y_teste)
        
        # normalization
        
        x_max_estacoes = torch.max(train_dataset.X[:, 0])
        x_max_cosmo = torch.max(train_dataset.X[:,1])
        x_max_gfs25 = torch.max(train_dataset.X[:,2])
        
        y_max_estacoes = torch.max(train_dataset.Y[:, 0])
        
        print(train_dataset.X.shape)
        x_max = torch.tensor([x_max_estacoes, x_max_cosmo, x_max_gfs25]).reshape([1,3,1,1,1])
        
        train_dataset.X = (train_dataset.X) / (x_max)
        val_dataset.X = (val_dataset.X) / (x_max)
        test_dataset.X = (test_dataset.X) / (x_max)
        
        train_dataset.Y = (train_dataset.Y) / (y_max_estacoes)
        val_dataset.Y = (val_dataset.Y) / (y_max_estacoes)
        test_dataset.Y = (test_dataset.Y) / (y_max_estacoes)
        
        if (self.config.verbose):
            print('[X_train] Shape:', train_dataset[0][0].shape)
            print('[y_train] Shape:', train_dataset[0][1].shape)
            print('[X_val] Shape:', val_dataset[0][0].shape)
            print('[y_val] Shape:', val_dataset[0][1].shape)
            print('[X_test] Shape:', test_dataset[0][0].shape)
            print('[y_test] Shape:', test_dataset[0][1].shape)
            print(f'Train on {len(train_dataset)} samples, validate on {len(val_dataset)} samples')
                        
        params = {'batch_size': self.config.batch, 
                  'num_workers': self.config.workers, 
                  'worker_init_fn': self.__init_seed}
        
        params_v = {'batch_size': self.config.batch, 
                  'num_workers': self.config.workers, 
                  'worker_init_fn': self.__init_seed}

        train_loader = DataLoader(dataset=train_dataset, shuffle=True,**params)
        val_loader = DataLoader(dataset=val_dataset, shuffle=False,**params_v)
        test_loader = DataLoader(dataset=test_dataset, shuffle=False, **params_v)
        
        models = {
            'stconvs2s-r': STConvS2S_R,
            'stconvs2s-c': STConvS2S_C,
            'convlstm': STConvLSTM,
            'predrnn': PredRNN,
            'mim': MIM,
            'conv2plus1d': Conv2Plus1D,
            'conv3d': Conv3D,
            'enc-dec3d': Endocer_Decoder3D,
            'ablation-stconvs2s-nocausalconstraint': AblationSTConvS2S_NoCausalConstraint,
            'ablation-stconvs2s-notemporal': AblationSTConvS2S_NoTemporal,
            'ablation-stconvs2s-r-nochannelincrease': AblationSTConvS2S_R_NoChannelIncrease,
            'ablation-stconvs2s-c-nochannelincrease': AblationSTConvS2S_C_NoChannelIncrease,
            'ablation-stconvs2s-r-inverted': AblationSTConvS2S_R_Inverted,
            'ablation-stconvs2s-c-inverted': AblationSTConvS2S_C_Inverted,
            'ablation-stconvs2s-r-notfactorized': AblationSTConvS2S_R_NotFactorized,
            'ablation-stconvs2s-c-notfactorized': AblationSTConvS2S_C_NotFactorized
        }
        if not(self.config.model in models):
            raise ValueError(f'{self.config.model} is not a valid model name. Choose between: {models.keys()}')
            quit()
            
        # Creating the model    
        model_bulder = models[self.config.model]
        model = model_bulder(train_dataset.X.shape, self.config.num_layers, self.config.hidden_dim, 
                             self.config.kernel_size, self.device, self.dropout_rate, self.y_step)
        
        # weights initialization
        train_mean = torch.mean(train_dataset.Y)
        model.apply(lambda m: self.weights_init(m, train_mean))
        
        model.to(self.device)
        mask = torch.from_numpy(xr.open_dataarray('data/mask.nc').values).float().to(self.device)
        criterion = CustomMSELoss(mask)
        #criterion = nn.MSELoss()
        opt_params = {'lr': self.config.learning_rate, 
                      'alpha': 0.99, 
                      'eps': 1e-6,
                      'weight_decay': 1e-1}
        optimizer = torch.optim.RMSprop(model.parameters(), **opt_params)
        util = Util(self.config.model, self.dataset_type, self.config.version, self.filename_prefix)
        
        train_info = {'train_time': 0}
        if self.config.pre_trained is None:
            train_info = self.__execute_learning(model, criterion, optimizer, train_loader,  val_loader, util) 
        
        train_min = torch.tensor([0.0]).reshape(1,1,1,1,1).to(self.device)
        train_max = y_max_estacoes.reshape(1,1,1,1,1).to(self.device)
        eval_info = self.__load_and_evaluate(model, criterion, optimizer, test_loader, train_min, train_max, 
                                             train_info['train_time'], util)

        if (torch.cuda.is_available()):
            torch.cuda.empty_cache()

        return {**train_info, **eval_info}


    def __execute_learning(self, model, criterion, optimizer, train_loader, val_loader, util):
        checkpoint_filename = util.get_checkpoint_filename()    
        trainer = Trainer(model, criterion, optimizer, train_loader, val_loader, self.config.epoch, 
                          self.device, util, self.config.verbose, self.config.patience, self.config.no_stop)
    
        start_timestamp = tm.time()
        # Training the model
        train_losses, val_losses = trainer.fit(checkpoint_filename, is_chirps=self.config.chirps)
        end_timestamp = tm.time()
        # Learning curve
        util.save_loss(train_losses, val_losses)
        util.plot([train_losses, val_losses], ['Training', 'Validation'], 'Epochs', 'Loss',
                  'Learning curve - ' + self.config.model.upper(), self.config.plot)

        train_time = end_timestamp - start_timestamp       
        print(f'\nTraining time: {util.to_readable_time(train_time)} [{train_time}]')
               
        return {'dataset': self.dataset_name,
                'dropout_rate': self.dropout_rate,
                'train_time': train_time
                }
                
    
    def __load_and_evaluate(self, model, criterion, optimizer, test_loader, train_min, train_max, train_time, util):  
        evaluator = Evaluator(model, criterion, optimizer, test_loader, train_min, train_max, self.device, util, self.y_step)
        if self.config.pre_trained is not None:
            # Load pre-trained model
            best_epoch, val_loss = evaluator.load_checkpoint(self.config.pre_trained, self.dataset_type, self.config.model)
        else:
            # Load model with minimal loss after training phase
            checkpoint_filename = util.get_checkpoint_filename() 
            best_epoch, val_loss = evaluator.load_checkpoint(checkpoint_filename)
        
        time_per_epochs = 0
        if not(self.config.no_stop): # Earling stopping during training
            time_per_epochs = train_time / (best_epoch + self.config.patience)
            print(f'Training time/epochs: {util.to_readable_time(time_per_epochs)} [{time_per_epochs}]')
        
        test_rmse, test_mae = evaluator.eval(is_chirps=self.config.chirps)
        print(f'Test MSE: {test_rmse:.4f}\nTest MAE: {test_mae:.4f}')
                        
        return {'best_epoch': best_epoch,
                'val_rmse': val_loss,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'train_time_epochs': time_per_epochs
                }
          
    def __define_seed(self, number):      
        if (~self.config.no_seed):
            # define a different seed in every iteration 
            seed = (number * 10) + 1000
            np.random.seed(seed)
            rd.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic=True
            
    def __init_seed(self, number):
        seed = (number * 10) + 1000
        np.random.seed(seed)
        
    def __get_dataset_file(self):
        dataset_file, dataset_name = None, None
        if self.config.dataset:
            dataset_file = 'data/' + self.config.dataset
            dataset_name = 'COR'
            
        return dataset_name, dataset_file
        
    def __get_dropout_rate(self):
        dropout_rates = {
            'predrnn': 0.5,
            'mim': 0.5
        }
        if self.config.model in dropout_rates:
            dropout_rate = dropout_rates[self.config.model] 
        else:
            dropout_rate = 0.

        return dropout_rate
    
    def weights_init(self, model, train_mean):
        if isinstance(model, nn.Conv3d):
            nn.init.kaiming_uniform_(model.weight)
            if model.bias is not None:
                print(f"MEAN: {train_mean}")
                nn.init.constant_(model.bias, train_mean)