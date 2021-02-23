import os
import random
import sys
from abc import ABC, abstractmethod

import click
import numpy as np
import pandas as pd
import torch
import yaml
from torch.optim import Optimizer

import callbacks
import dataset
from dataset import Dataset, KFoldCV
from model import CleavagePredictor


class TrainingConfigurator(ABC):
    '''
    contains training parameters network architecture
    '''
    epochs: int = None
    output_dir: str = None
    cv_do: bool = None
    cv_repetitions: int = None
    cv_folds: int = None
    batch_size: int = None
    max_epochs: int = None
    dataset_path: str = None
    sample: int = None
    _device: str = 'auto'

    @property
    def device(self):
        if self._device == 'auto':
            return 'cuda:0' if torch.cuda.is_available() else 'cpu'
        else:
            return self._device

    @device.setter
    def set_device(self, device):
        self._device = device

    @abstractmethod
    def get_predictor(self) -> CleavagePredictor:
        '''
        return a new predictor. in case of cross-validation,
        this method will be called for each fold
        '''
        pass

def standard_split(data, val_size):
    val_size = int(val_size * len(data))
    val_data = data.sample(val_size)
    train_data = data.iloc[~data.index.isin(val_data.index)] 
    return train_data, val_data

class Trainer:
    def __init__(self, configurator: TrainingConfigurator):
        self._config = configurator

    def run(self):

        all_data = pd.read_csv(self._config.dataset_path)
        try: all_data = all_data[all_data['Start_Position'] != 'Homo sapiens']
        except: all_data = all_data
        try: all_data = dataset.label_bags(all_data.to_numpy(), self._config.negatives_generator, self._config.number_of_negatives, self._config.nterm, self._config.aa_left, self._config.aa_right)
        except: all_data = all_data#all_data = all_data[0:5000]
        sample = self._config.sample
        if sample is not None:
            all_data = all_data.sample(sample)
        print(' number of rows :', len(all_data))


        if not self._config.cv_do:
            if self._config.use_split: loss = self.use_split(all_data)
            else: loss = self.run_simple_split(all_data)
        else:
            loss = self.run_repeated_crossvalidation(all_data)
            #except: loss = 1

        with open(f'{self._config.output_dir}/result', 'w') as f:
            f.write(str(loss) + '\n')

    def run_simple_split(self, all_data):
        print('runs simple split')
        train_data, val_data = standard_split(all_data, 0.25)
        print('train_data:', len(train_data), 'val_data:', len(val_data))
        test_acc, test_auc, test_mse, test_xen, test_ano, test_f1, test_mcc, best = self.do_training(train_data, val_data, None)
        return best

    def use_split(self, all_data):
        if self._config.negatives_generator == 'shuffled_negatives' and self._config.mhc == 1 and self._config.aa_left == 3:
            train_data = pd.read_csv('./data/mhc1_shuff_train_3.csv')
            val_data = pd.read_csv('./data/mhc1_shuff_test_3.csv')
        if self._config.negatives_generator != 'shuffled_negatives' and self._config.mhc == 1 and self._config.aa_left == 3:
            train_data = pd.read_csv('./data/mhc1_shift_train_3.csv')
            val_data = pd.read_csv('./data/mhc1_shift_test_3.csv')
        if self._config.negatives_generator == 'shuffled_negatives' and self._config.mhc == 1 and self._config.aa_left == 7:
            train_data = pd.read_csv('./data/mhc1_shuff_train_7.csv')
            val_data = pd.read_csv('./data/mhc1_shuff_test_7.csv')
        if self._config.negatives_generator != 'shuffled_negatives' and self._config.mhc == 1 and self._config.aa_left == 7:
            train_data = pd.read_csv('./data/mhc1_shift_train_7.csv')
            al_data = pd.read_csv('./data/mhc1_shift_test_7.csv')
        if self._config.negatives_generator == 'shuffled_negatives' and self._config.mhc == 1 and self._config.aa_left == 15:
            train_data = pd.read_csv('./data/mhc1_shuff_train.csv')
            val_data = pd.read_csv('./data/mhc1_shuff_test.csv')
        if self._config.negatives_generator != 'shuffled_negatives' and self._config.mhc == 1 and self._config.aa_left == 15:
            train_data = pd.read_csv('./data/mhc1_shift_train.csv')
            al_data = pd.read_csv('./data/mhc1_shift_test.csv')
        if self._config.negatives_generator == 'shuffled_negatives' and self._config.mhc == 2 and self._config.aa_left == 15:
            train_data = pd.read_csv('./data/mhc2_shuff_train.csv')
            val_data = pd.read_csv('./data/mhc2_shuff_test.csv')
        if self._config.negatives_generator != 'shuffled_negatives' and self._config.mhc == 2 and self._config.aa_left == 15:
            train_data = pd.read_csv('./data/mhc2_shift_train.csv')
            al_data = pd.read_csv('./data/mhc2_shift_test.csv')
        if self._config.mhc == 3 and self._config.aa_left == 3:
            train_data = pd.read_csv('./data/vitro_Calis_I_3_3.csv')   
            val_data = pd.read_csv('./data/vitro_Calis_I_3_3.csv')
        if self._config.mhc == 3 and self._config.aa_left == 7:
            train_data = pd.read_csv('./data/vitro_Calis_I_7_7.csv')
            val_data = pd.read_csv('./data/vitro_Calis_I_7_7.csv')
        if self._config.mhc == 3 and self._config.aa_left == 15:
            train_data = pd.read_csv('./data/vitro_Calis_I_15_15.csv')
            val_data = pd.read_csv('./data/vitro_Calis_I_15_15.csv')
        if self._config.mhc == 4 and self._config.aa_left == 3:
            train_data = pd.read_csv('./data/vitro_Calis_C_3_3.csv')  
            val_data = pd.read_csv('./data/vitro_Calis_C_3_3.csv')
        if self._config.mhc == 4 and self._config.aa_left == 7:
            train_data = pd.read_csv('./data/vitro_Calis_C_7_7.csv')
            val_data = pd.read_csv('./data/vitro_Calis_C_7_7.csv')
        if self._config.mhc == 4 and self._config.aa_left == 15:
            train_data = pd.read_csv('./data/vitro_Calis_C_15_15.csv')
            val_data = pd.read_csv('./data/vitro_Calis_C_15_15.csv')
        if self._config.mhc == 5 and self._config.aa_left == 3:
            train_data = pd.read_csv('./data/vitro_all_I_3_3.csv')
            val_data = pd.read_csv('./data/vitro_all_I_3_3.csv')
        if self._config.mhc == 5 and self._config.aa_left == 7:
            train_data = pd.read_csv('./data/vitro_all_I_7_7.csv')
            val_data = pd.read_csv('./data/vitro_all_I_7_7.csv')
        if self._config.mhc == 5 and self._config.aa_left == 15:
            train_data = pd.read_csv('./data/vitro_all_I_15_15.csv')
            val_data = pd.read_csv('./data/vitro_all_I_15_15.csv')
        if self._config.mhc == 6 and self._config.aa_left == 3:
            train_data = pd.read_csv('./data/vitro_all_C_3_3.csv')
            val_data = pd.read_csv('./data/vitro_all_C_3_3.csv')
        if self._config.mhc == 6 and self._config.aa_left == 7:
            train_data = pd.read_csv('./data/vitro_all_C_7_7.csv')
            val_data = pd.read_csv('./data/vitro_all_C_7_7.csv')
        if self._config.mhc == 6 and self._config.aa_left == 15:
            train_data = pd.read_csv('./data/vitro_all_C_15_15.csv')
            val_data = pd.read_csv('./data/vitro_all_C_15_15.csv')

        print('train_data:', len(train_data), 'val_data:', len(val_data))
        test_acc, test_auc, test_mse, test_xen, test_ano, test_f1, test_mcc, best = self.do_training(train_data, val_data, None)
        return best

    def run_repeated_crossvalidation(self, all_data):
        rep_scores = {}
        for rep in range(self._config.cv_repetitions):
            folds_scores = {}
            kfold = KFoldCV(self._config.cv_folds, seed=self._config.cv_seed)
            for i, (train_mask, test_mask) in enumerate(kfold.split(all_data)):
                test_data = all_data[test_mask]
                train_data, val_data = standard_split(all_data[train_mask], 0.2)
                if self._config.aa_left == 3:
                     vitro_c = pd.read_csv('./data/vitro_Calis_C_3_3.csv')
                     vitro_i = pd.read_csv('./data/vitro_Calis_I_3_3.csv')
                if self._config.aa_left == 7:
                     vitro_c = pd.read_csv('./data/vitro_Calis_C_7_7.csv')
                     vitro_i = pd.read_csv('./data/vitro_Calis_I_7_7.csv')
                if self._config.aa_left == 15:
                     vitro_c = pd.read_csv('./data/vitro_Calis_C_15_15.csv')
                     vitro_i = pd.read_csv('./data/vitro_Calis_I_15_15.csv')
                print(f'Starting fold {i} of repetition {rep}')
                print(f'{len(train_data)} samples for training, '
                      f'{len(val_data)} for validation '
                      f'and {sum(test_mask)} for testing')

                test_mse, test_xen, test_acc, test_auc, test_ano, test_f1, test_mcc, best = self.do_training(
                    train_data, val_data, test_data, vitro_c, vitro_i
                )
                #except: test_mse, test_xen, test_acc, test_auc, test_ano, test_f1, test_mcc, best = 1, 1, 0, 0, 0, 0, 0, 0
                folds_scores[f'fold-{i}'] = {
                    'crossent': test_xen, 'auc': test_auc, 'acc': test_acc, 'mse': test_mse, 'ano': test_ano, 'f1': test_f1, 'mcc': test_mcc, 'best': best
                }
            rep_scores[f'rep-{rep}'] = folds_scores

        with open(f'{self._config.output_dir}/cv-report.yaml', 'w') as f:
            yaml.dump(rep_scores, f)

        
        avg_combined_loss = -1 * np.mean([
            
            
            fold['mcc']
            for fold_scores in rep_scores.values()
            for fold in fold_scores.values()
        ])

        return avg_combined_loss
        

    def do_training(self, train_data, vali_data, test_data, vitro_c=None, vitro_i=None):
        train_dataset = Dataset(train_data, batch_size=self._config.batch_size,
                                shuffle=True, device=self._config.device, aa_left = self._config.aa_left, aa_right = self._config.aa_right)
        val_dataset = Dataset(vali_data, batch_size= self._config.batch_size,
                              device=self._config.device, aa_left = self._config.aa_left, aa_right = self._config.aa_right)
        if test_data is not None:
            test_dataset = Dataset(test_data, batch_size= self._config.batch_size,
                                   device=self._config.device, aa_left = self._config.aa_left, aa_right = self._config.aa_right)
        else:
            print('Warning: metrics refer to validation data')
            test_dataset = val_dataset


        predictor = self._config.get_predictor()
        predictor.fit(train_dataset, self._config.epochs, val_dataset, callbacks=[
            callbacks.EarlyStopping(metric='mcc', patience=10, min_delta=0.005),
            callbacks.ModelCheckpoint(
                metric='mcc', fname=f'{self._config.output_dir}/state_dict.pto'),
            callbacks.TrackParameter(parameters = ['loss', 'acc', 'ano', 'mcc', 'auc', 'mse', 'noise_layer'], fname = f'{self._config.output_dir}/summary')
        ], start_noise = self._config.start_noise)




        test_mse, test_xen, test_acc, test_auc, test_ano, test_f1, test_mcc, best = predictor.test_best(test_dataset,0,path=f'{self._config.output_dir}/state_dict.pto')
        print(
            f'ACC: {test_acc:.3f} - AUC: {test_auc:.3f} - MSE: {test_mse:.3f} - XEN: {test_xen:.3f} - ANO: {test_ano:.3f} - F1: {test_f1:.3f} - MCC: {test_mcc:.3f} - BEST: {best:.3f}\n')
        print('ACC: {:.3f} - AUC: {:.3f} - MSE: {:.3f} - XEN: {:.3f} - ANO: {:.3f} - F1: {:.3f} - MCC: {:.3f} - BEST: {:.3f}\n'.format(
            test_acc, test_auc, test_mse, test_xen, test_ano, test_f1, test_mcc, best
        ))

        return test_mse, test_xen, test_acc, test_auc, test_ano, test_f1, test_mcc, best


#@click.command()
#@click.argument('configurator-path', type=click.File('r'))
#@click.option('-e', '--epochs', type=int, help='Override number of epochs')
#@click.option('-o', '--output-dir', type=click.Path(), help='Override output directory')
def cli(configurator_path, epochs, output_dir):
    defs = {}
    src = configurator_path.read()
    exec(src, defs)

    config = defs.get('configurator')
    if config is None:
        print('ERROR: make sure the configurator module exposes a variable')
        print('ERROR: named "configurator" that inherits from TrainingConfigurator')
        print('ERROR: keys defined in the imported module:', ', '.join(defs.keys()))
        sys.exit(-1)

    if epochs is not None:
        print(f'Updating training epochs to {epochs}')
        config.epochs = epochs

    if output_dir is not None:
        print(f'Updating output directory to {output_dir}')
        config.output_dir = output_dir

    Trainer(config).run()


#if __name__ == '__main__':
#     cli()
