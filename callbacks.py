import os
from queue import deque

import torch


class Callback:
    '''
    base class for callbacks used during fitting of the model.
    return True from any method to stop training
    '''

    def on_epoch_start(self, model, epoch):
        pass

    def on_epoch_end(self, model, epoch, keep_noise_constant, validation_mse, validation_crossent, validation_acc, validation_auc, validation_ano,validation_f1, validation_mcc, validation_best):
        pass

    def on_batch_start(self, model, epoch, batch, batch_data):
        pass

    def on_batch_end(self, model, epoch, batch, batch_data, predictions):
        pass

    def on_fitting_batch_start(self, model, epoch, batch, batch_data):
        pass

    def on_fitting_batch_end(self, model, epoch, batch, batch_data, predictions, bar_length, accuracy):
        pass

    def on_validating_batch_start(self, model, epoch, batch, batch_data):
        pass

    def on_validating_epoch_end(self, model, epoch, batch, batch_data, predictions, validation_mse, validation_crossent, validation_acc, validation_auc, validation_ano,validation_f1, validation_mcc,validation_best):
        pass

    def on_validating_batch_end(self, model, epoch, batch, predictions, bar_length):
        pass

    def on_validation_end(self, validation_mse, validation_crossent, validation_acc, validation_auc, validation_ano,validation_f1, validation_mcc):
        pass

class EarlyStopping(Callback):
    def __init__(self, patience, min_delta=1e-3, metric='mcc'):
        self._metric = metric
        self._patience = patience
        self._min_delta = min_delta
        self._metrics = deque(maxlen=self._patience)
        self._worse_values = deque(maxlen=self._patience)
        self._best_value = None

    def on_epoch_end(self, model, epoch, keep_noise_constant, validation_mse, validation_crossent, validation_acc, validation_auc, validation_ano, validation_f1, validation_mcc,validation_best):
        if self._metric == 'auc':
            metric, smaller = validation_auc, False
        elif self._metric == 'mse':
            metric, smaller = validation_mse, True
        elif self._metric == 'acc':
            metric, smaller = validation_acc, False
        elif self._metric == 'crossent':
            metric, smaller = validation_crossent, True
        elif self._metric == 'ano':
            metric, smaller = validation_ano, False
        elif self._metric == 'f1':
            metric, smaller = validation_f1, False
        elif self._metric == 'mcc':
            metric, smaller = validation_mcc, False
        else:
            raise ValueError('unknown metric, use either ano, f1, mcc, auc, mse, crossent or acc')

        if metric is None:
            raise ValueError('you must use a validation dataset with this callback')
 
        if (
            self._best_value is None
        ) or (
            smaller and metric < self._best_value
        ) or (
            not smaller and metric > self._best_value
        ):
           self._best_value = metric
           self._worse_values = deque(maxlen=self._patience)

        if (
            self._best_value is not None
        ) or (
            smaller and 0.9 *  metric > self._best_value
        ) or (
            not smaller and 1.1 * metric < self._best_value
        ): 
           self._worse_values.append(metric)


        self._metrics.append(metric)
        
        if len(self._metrics) == self._patience:
            sm, lg = min(self._metrics), max(self._metrics)
            if lg - sm < self._min_delta or len(self._worse_values) ==0.5*self._patience:
                self._patience =  self._patience
                self._worse_values = deque(maxlen=self._patience)
                self._metrics = deque(maxlen=self._patience)
                return True
            else:                    
                return False
 
         

class ModelCheckpoint(Callback):
    def __init__(self, fname, metric):
        self._metric = metric
        self._best_value = None
        self._fname = fname
        self._fname_noise = fname[slice(0,-4)] + '_noise.pto'
        self._fname_no_noise = fname[slice(0,-4)] + '_no_noise.pto'
        self._best_value_noise = None
        self._best_value_no_noise = None
     

        output_dir, _ = os.path.split(fname)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

    def on_epoch_end(self, model, epoch, keep_noise_constant, validation_mse, validation_crossent, validation_acc, validation_auc, validation_ano, validation_f1, validation_mcc,validation_best):
        if self._metric == 'auc':
            metric, smaller = validation_auc, False
        elif self._metric == 'mse':
            metric, smaller = validation_mse, True
        elif self._metric == 'acc':
            metric, smaller = validation_acc, False
        elif self._metric == 'crossent':
            metric, smaller = validation_crossent, True
        elif self._metric == 'ano':
            metric, smaller = validation_ano, False
        elif self._metric == 'f1':
            metric, smaller = validation_f1, False
        elif self._metric == 'mcc':
            metric, smaller = validation_mcc, False
        else:
            raise ValueError('unknown metric, use either auc, f1, mcc, ano, mse, crossent or acc')

        if metric is None:
            raise ValueError('you must use a validation dataset with this callback')

        if (
            self._best_value is None
        ) or (
            smaller and metric < self._best_value
        ) or (
            not smaller and metric > self._best_value
        ):
           #print(self._best_value, 'best value', metric, 'metric')
           self._best_value = metric
           #print(self._best_value, 'best value')
           torch.save(model._nnet.state_dict(), self._fname.format(epoch=epoch))

        if keep_noise_constant and ((
            self._best_value_no_noise is None
        ) or (
            smaller and metric < self._best_value_no_noise
        ) or (
            not smaller and metric > self._best_value_no_noise
        )):
           self._best_value_no_noise = metric
           torch.save(model._nnet.state_dict(), self._fname_no_noise.format(epoch=epoch))

        if not keep_noise_constant and ((
            self._best_value_noise is None
        ) or (
            smaller and metric < self._best_value_noise
        ) or (
            not smaller and metric > self._best_value_noise
        )):
           self._best_value_noise = metric
           torch.save(model._nnet.state_dict(), self._fname_noise.format(epoch=epoch))

    def on_validation_end(self,validation_mse, validation_crossent, validation_acc, validation_auc, validation_ano,validation_f1, validation_mcc):
        if self._metric == 'auc':
            metric, smaller = validation_auc, False
        elif self._metric == 'mse':
            metric, smaller = validation_mse, True
        elif self._metric == 'acc':
            metric, smaller = validation_acc, False
        elif self._metric == 'crossent':
            metric, smaller = validation_crossent, True
        elif self._metric == 'ano':
            metric, smaller = validation_ano, False
        elif self._metric == 'f1':
            metric, smaller = validation_f1, False
        elif self._metric == 'mcc':
            metric, smaller = validation_mcc, False
        else:
            raise ValueError('unknown metric, use either auc, f1, mcc, ano, mse, crossent or acc')

        if metric is None:
            raise ValueError('you must use a validation dataset with this callback')

        if self._best_value is None:
           return 0.0
        elif smaller:
           return min(metric,self._best_value)
        else:
           return max(metric,self._best_value)



class TrackParameter(Callback):
    def __init__(self, parameters, fname, n = 100):
        self._parameters = parameters
        self._n = n
        from torch.utils.tensorboard import SummaryWriter
        output_dir, _ = os.path.split(fname)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        self._writer = SummaryWriter(output_dir)

        self._running_loss_training = None
        self._running_noise_training = None  
        self._running_loss_validation = None
        self._running_acc_training = None 
          
        

    def on_fitting_batch_end(self, model, epoch, batch, batch_data, predictions, bar_length, accuracy):
        params, names = [], []
        if 'loss' in self._parameters:
            param, name = predictions.loss, 'training_loss'
            params.append(param), names.append(name)
        if 'acc' in self._parameters:
            param, name = accuracy, 'training_acc'
            params.append(param), names.append(name)
        if 'noise_layer' in self._parameters:
            param, name = torch.sigmoid(model._nnet.noise_layer.b.weight.data), 'noise_layer'
            params.append(param), names.append(name)

        for param, name in zip(params, names):
          if name == 'training_loss' and self._running_loss_training is None:
             self._running_loss_training = param.item()
          if name == 'training_loss' and self._running_loss_training is not None:
             self._running_loss_training += param.item()
          if name == 'training_acc' and self._running_acc_training is None:
             self._running_acc_training = param.item()
          if name == 'training_acc' and self._running_acc_training is not None:
             self._running_acc_training += param.item()
          if name == 'noise_layer' and self._running_noise_training is None:
             self._running_noise_training = param.item()
          if name == 'noise_layer' and self._running_noise_training is not None:
             self._running_noise_training += param.item()
          
          if batch % self._n == self._n-1:
              if name == 'training_loss':
                  self._writer.add_scalar(name,self._running_loss_training / self._n,epoch * bar_length + batch)
                  self._running_loss_training = 0.0
              if name == 'noise_layer':
                  self._writer.add_scalar(name,self._running_noise_training / self._n,epoch * bar_length + batch)
                  self._running_noise_training = 0.0
              if name == 'training_acc':
                  self._writer.add_scalar(name,self._running_acc_training / self._n,epoch * bar_length + batch)
                  self._running_acc_training = 0.0

    def on_validating_batch_end(self, model, epoch, batch, predictions, bar_length):
        if 'loss' in self._parameters:
          name, param = 'validation_loss', predictions.loss
          if name == 'validation_loss' and self._running_loss_validation is None:
             self._running_loss_validation = param.item()
          if name == 'validation_loss' and self._running_loss_validation is not None:
             self._running_loss_validation += param.item()
          if batch % self._n == self._n-1:
              if name == 'validation_loss':
                  self._writer.add_scalar(name,self._running_loss_validation / self._n,epoch * bar_length + batch)
                  self._running_loss_validation = 0.0


    def on_epoch_end(self, model, epoch, keep_noise_constant, validation_mse, validation_crossent, validation_acc, validation_auc, validation_ano,validation_f1, validation_mcc, validation_best):
        if 'mcc' in self._parameters:
            self._writer.add_scalar('mcc', validation_mcc,epoch)
        if 'acc' in self._parameters:
            self._writer.add_scalar('acc', validation_acc,epoch)
        if 'auc' in self._parameters:
            self._writer.add_scalar('auc', validation_auc,epoch)
        if 'ano' in self._parameters:
            self._writer.add_scalar('ano', validation_ano,epoch)
        if 'mse' in self._parameters:
            self._writer.add_scalar('mse', validation_mse,epoch)
        if 'xen' in self._parameters:
            self._writer.add_scalar('xen', validation_xen,epoch)
        if 'f1' in self._parameters:
            self._writer.add_scalar('f1', validation_f1,epoch)

