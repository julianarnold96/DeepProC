import abc
import dataclasses
import math
import random
from typing import Union

import numpy as np
import torch
from sklearn import metrics
from torch import nn
from torch.distributions import Beta
from tqdm.auto import tqdm, trange
from dataclasses import dataclass

import dataset
from utils import Smoothie

@dataclass
class Predictions:
    alphas: Union[np.array, torch.Tensor]
    logits: Union[np.array, torch.Tensor]
    loss: Union[np.array, torch.Tensor]

    def detach(self):

        if self.loss is not None:
            self.loss = self.loss.detach().cpu().numpy()
        self.alphas = self.alphas.detach().cpu().numpy()
        self.logits = self.logits.detach().cpu().numpy()


def anomaly_metric(y_true, y_pred):
  if sum(y_pred) == 0:
     return 0
  else:
     p = metrics.precision_score(y_true=y_true, y_pred=y_pred)
     r = metrics.recall_score(y_true=y_true, y_pred=y_pred) 
     y1 = sum(y_true)
     ytot = len(y_true)
     y1_share = y1/ytot
     if y1 > 0: 
        score = (p*r)/y1_share
        return score
     else: 
        return 0

class CleavagePredictor:
    '''
    wraps the actual network, providing facilities to train and evaluate
    '''

    def __init__(self, network, optimizer):
        self._nnet = network
        self._optimizer = optimizer

    def predict_batch(self, batch_data):
        return self._nnet(
            batch_data.mask, batch_data.proteins, batch_data.labels
        )

    def _fit_batch(self, batch_data):
        predictions = self.predict_batch(batch_data)
        self._optimizer.zero_grad()
        if torch.sum(torch.isnan(predictions.loss))>0:
          print(self._nnet) 
          for name, param in self._nnet.named_parameters():
             print(name, param.data.shape,torch.sum(torch.abs(param.data)),param.data)
        predictions.loss.backward()
        torch.nn.utils.clip_grad_norm_(self._nnet.parameters(), 0.25)
        self._optimizer.step()
        if self._keep_noise_constant:
            #print(self._keep_noise_constant, 'in fit', self._nnet.noise_layer.b.weight.data, 'bf') 
            self._nnet.noise_layer.b.weight.data.fill_(-4.6)
        #print(self._keep_noise_constant, 'in fit', self._nnet.noise_layer.b.weight.data, 'af') 

        return predictions

    def _print_params(self):
        for name, param in self._nnet.named_parameters():
            print(name, param.data)

    def _fit_one_epoch(self, epoch, train_dataset):
        self._nnet.train()
        self._stop_training = self._stop_training or any(
            cb.on_epoch_start(self, epoch) for cb in self._callbacks
        )

        smoothie = Smoothie()
        bar = tqdm(train_dataset, leave=False)
        for i, batch_data in enumerate(bar):
            self._stop_training = self._stop_training or any(
                cb.on_batch_start(self, epoch, i, batch_data) for cb in self._callbacks
            )

            predictions = self._fit_batch(batch_data)
            predictions.detach()

            self._stop_training = self._stop_training or any(
                cb.on_batch_end(self, epoch, i, batch_data, predictions)
                for cb in self._callbacks
            )

            #if np.sum(np.isnan(predictions.loss))>0:
            #     for name, param in self._nnet.named_parameters():
            #         print(name, param.data)
            assert predictions.loss is not None, self._print_params
            assert np.isfinite(predictions.loss), print(predictions.loss, predictions.loss is None)

            accuracy = np.mean([
                true == int(pred > 0.5)
                for true, pred in zip(batch_data.labels.cpu(), predictions.logits)
                if np.isfinite(true)
            ])

            self._stop_training = self._stop_training or any(
                cb.on_fitting_batch_end(self, epoch, i, batch_data, predictions, len(bar), accuracy) for cb in self._callbacks
            )

            smoothie.update(loss=predictions.loss,
                            acc=accuracy)
            bar.set_description(
                'LO: {loss:.3f} - AC: {acc:.3f}'
                ''.format(**smoothie.value())
            )

    def _validate_end_of_epoch(self, epoch, validation_dataset):
        if validation_dataset is None:
            self._stop_training = self._stop_training or any(
                cb.on_epoch_end(self, epoch, self._keep_noise_constant, None, None, None, None, None, None, None, None) for cb in self._callbacks
            )
            return

        val_mse, val_xen, val_acc, val_auc, val_ano, val_f1, val_mcc, val_best = self.validate(validation_dataset, epoch)
        
        self._stop_training = self._stop_training or any(
            cb.on_epoch_end(self, epoch,self._keep_noise_constant, val_mse, val_xen, val_acc, val_auc, val_ano, val_f1, val_mcc, val_best)
            for cb in self._callbacks
        )

        return val_mse, val_xen, val_acc, val_auc, val_ano, val_f1, val_mcc, val_best

    def validate(self, validation_dataset, epoch):
        self._nnet.eval()
        predictions, observations, squared_errors = [], [], []
        with torch.no_grad():
            bar_val = tqdm(validation_dataset, leave=False)
            for i, batch_data in enumerate(bar_val):
                predictions_data = self.predict_batch(batch_data)
                predictions_data.detach()
                self._stop_training = self._stop_training or any(
                cb.on_validating_batch_end(self, epoch, i, predictions_data, len(bar_val)) for cb in self._callbacks
                )
                for obs, pred in zip(batch_data.labels.cpu(), predictions_data.logits):
                    #print(obs,pred)
                    if np.isfinite(obs):
                        predictions.append(pred)
                        observations.append(obs)

                squared_errors.extend([
                    (obs - 1/(1+np.exp(-np.array(pred))))**2
                    for obs, pred in zip(
                        batch_data.labels.cpu(),
                        predictions_data.logits,
                    ) if np.isfinite(obs)
                ])

        
        predictions = 1 / (1 + np.exp(-np.array(predictions)))
        val_xen = -np.mean(observations * np.log(predictions))
        val_auc = metrics.roc_auc_score(observations, predictions)
        val_acc = metrics.accuracy_score(observations, [p > 0.5 for p in predictions])
        val_mse = np.mean(squared_errors)
        rounded_predictions = np.round(predictions)
        val_ano = anomaly_metric(observations, rounded_predictions)
        val_f1 = metrics.f1_score(observations, rounded_predictions)
        if sum(observations) * sum(rounded_predictions) != 0:
            val_mcc = metrics.matthews_corrcoef(observations, rounded_predictions)
        else:
            val_mcc = 0
        
        val_best = self._callbacks[1].on_validation_end(val_mse, val_xen, val_acc, val_auc, val_ano, val_f1, val_mcc)

        return val_mse, val_xen, val_acc, val_auc, val_ano, val_f1, val_mcc, val_best

    def test_best(self, validation_dataset, epoch, path):
        self._nnet.load_state_dict(torch.load(path))      
        self._nnet.eval()
        predictions, observations, squared_errors = [], [], []
        with torch.no_grad():
            bar_test = tqdm(validation_dataset, leave=False)
            for i, batch_data in enumerate(bar_test):
                predictions_data = self.predict_batch(batch_data)
                predictions_data.detach()
                self._stop_training = self._stop_training or any(
                cb.on_validating_batch_end(self, epoch, i, predictions_data, len(bar_test)) for cb in self._callbacks
                )
                for obs, pred in zip(batch_data.labels.cpu(), predictions_data.logits):
                    if np.isfinite(obs):
                        predictions.append(pred)
                        observations.append(obs)

                squared_errors.extend([
                    (obs - 1/(1+np.exp(-np.array(pred))))**2
                    for obs, pred in zip(
                        batch_data.labels.cpu(),
                        predictions_data.logits,
                    ) if np.isfinite(obs)
                ])

        predictions = 1 / (1 + np.exp(-np.array(predictions)))
        test_xen = -np.mean(observations * np.log(predictions))
        test_auc = metrics.roc_auc_score(observations, predictions)
        test_acc = metrics.accuracy_score(observations, [p > 0.5 for p in predictions])
        test_mse = np.mean(squared_errors)
        rounded_predictions = np.round(predictions)
        test_ano = anomaly_metric(observations, rounded_predictions)
        test_f1 = metrics.f1_score(observations, rounded_predictions)
        if sum(observations) * sum(rounded_predictions) != 0:
            test_mcc = metrics.matthews_corrcoef(observations, rounded_predictions)
        else:
            test_mcc = 0

        best = self._callbacks[1].on_validation_end(test_mse, test_xen, test_acc, test_auc, test_ano, test_f1, test_mcc)

        return test_mse, test_xen, test_acc, test_auc, test_ano, test_f1, test_mcc, best

    def fit(self, train_dataset, epochs, validation_dataset=None, callbacks=None, start_noise = None):
        # start_noise sets the epoch after which the noise layer gets updated, if left at None it gets activated after first convergence
        
        self._callbacks = callbacks or []
        self._keep_noise_constant = True     
        bar = trange(epochs)
        for epoch in bar:
            # will be set by callbacks
            self._stop_training = False

            self._fit_one_epoch(epoch, train_dataset)
            val_mse, val_xen, val_acc, val_auc, val_ano, val_f1, val_mcc, val_best = self._validate_end_of_epoch(
                epoch, validation_dataset)
          
            bar.set_description(
                'Val. AUC: {:.3f} - ANO: {:.3f} - F1: {:.3f} - MCC: {:.3f} - MSE: {:.3f} - ACC: {:.3f} - XEN: {:.3f} - BEST: {:.3f}'.format(
                    val_auc, val_ano, val_f1, val_mcc, val_mse, val_acc, val_xen, val_best
                )
            )
            print('epoch:',epoch)
            print('noise (b, sigmoid(b)):', self._nnet.noise_layer.b.weight.data, torch.sigmoid(self._nnet.noise_layer.b.weight.data))
            print('amino acids left:', torch.sum(torch.round(self._nnet.length_regulator.weights.weight.data[0][0:15])))
            print('amino acids right:',torch.sum(torch.round(self._nnet.length_regulator.weights.weight.data[0][15:30])))
            if start_noise is None:
                if self._stop_training and self._keep_noise_constant:
                     self._keep_noise_constant = False
                     self._stop_training = False
                     print('Noise Layer activated in epoch:', epoch)
                elif self._stop_training and not self._keep_noise_constant:
                     print('Converged in epoch:', epoch )
                     break
            elif epoch == start_noise:
                self._keep_noise_constant = False

class PositionalLengthRegularization(nn.Module):

      """
    
      one weight (initialized at 1) is assigned to each position
      the amino acid vector is multiplied with the rounded weights before entering the network
      as regularization, length_reg(self) is added to the loss
      here, the weights are multiplied with a 'punishment vector', set up as follows: 
      ( punishment_weight_left * (aa_left, aa_left - 1, ..., 3, 2, 1), punishment_weight_right * (1, 2, 3, ..., aa_right - 1, aa_right))

      the default values should reduce the 30 amino acids to 8 left of the ct and 1 right of the ct within the first epoch


      """

      def __init__(self,device, punishment_weight_left = 0.112, punishment_weight_right = 0.525, aa_left = 15, aa_right = 15):
        super().__init__()
        self.device = device
        self.aa_left = aa_left
        self.aa_right = aa_right
        self.weights = nn.Linear(self.aa_left+self.aa_right,1, bias = False)
        self.weights.weight.data.fill_(1)
        self.left_half = punishment_weight_left * torch.cat([torch.arange(15.-(15-self.aa_left), 0., -1)]).to(self.device)
        self.right_half = punishment_weight_right * torch.cat([torch.arange(1., 16.-(15-self.aa_right), 1)]).to(self.device)

      def length_reg(self):
        length_reg = self.weights.weight.view(self.aa_left+self.aa_right)
        punish_outer_aa = (torch.cat([self.left_half, self.right_half])) + 1
        length_reg = length_reg * punish_outer_aa       
        return torch.sum(torch.abs(1-(torch.abs((length_reg).view(self.aa_left+self.aa_right)))))

      def forward(self,x):
        length_reg = self.weights.weight.view(self.aa_left+self.aa_right)
        length_reg = torch.round(length_reg) 
        return x * length_reg.view(self.aa_left+self.aa_right)

class NoLengthRegularization(nn.Module):

      def __init__(self,device, punishment_weight_left = 0.075, punishment_weight_right = 0.35, aa_left = 15, aa_right = 15):
        super().__init__()
        
      def length_reg(self):
        return 0

      def forward(self,x):
        return x


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class SkipConnection(nn.Module):
    def __init__(self, inner, shape_residual_input=None):
        super().__init__()
        if shape_residual_input is None:
            self._shape_residual_input = nn.Identity()
        else:
            self._shape_residual_input = shape_residual_input
        self._inner = inner

    def forward(self, x):
        xs = self._shape_residual_input(x)
        return 0.25 * xs + self._inner(x)


class InceptionLayer(nn.Module):
    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.c1 = nn.Conv1d(in_chans, out_chans, kernel_size=1)
        self.c3 = nn.Conv1d(in_chans, out_chans, kernel_size=3, padding=1)
        self.c5 = nn.Conv1d(in_chans, out_chans, kernel_size=5, padding=2)
        self.ce = nn.Conv1d(3 * out_chans, out_chans, kernel_size=1)
        #self.bn = nn.BatchNorm1d(out_chans)

    def forward(self, x):
        x = torch.cat([
            self.c1(x), self.c3(x), self.c5(x),
        ], dim=1)
        return self.ce(x)

class EncoderBlock(abc.ABC):
    @abc.abstractmethod
    def build_block(self, index: int, prev_chans: int) -> (nn.Module, int):
        '''
        return the block at the given index and the new number of channels.
        '''

class VggBlock(EncoderBlock):
    def __init__(self, base=32, factor=2, count=2):
        self._factor = factor
        self._count = count
        self._base = base

    def build_block(self, index, prev_chans):
        layers = []
        this_chans = int(self._base * self._factor ** index)
        for _ in range(self._count):
            layers.extend([
                nn.Conv1d(prev_chans, this_chans, kernel_size=3, padding=1),
                nn.BatchNorm1d(this_chans),
                nn.LeakyReLU(),
            ])
            prev_chans = this_chans
        return nn.Sequential(*layers), this_chans


class InceptionBlock(EncoderBlock):
    def __init__(self, base=32, factor=2):
        self._factor = factor
        self._base = base

    def build_block(self, index, prev_chans):
        this_chans = int(self._base * self._factor ** index)
        return InceptionLayer(prev_chans, this_chans), this_chans


class PeptideEncoder(nn.Sequential):
    def __init__(self, aa_encoder, device, length_regulator, base=32, factor=2, blocks=2,
                 block_size=2, block_template=VggBlock(), out_dim=128,
                 dropout=0.5, skips_blocks=True):
        self._out_dim = out_dim
        self.device = device

        layers = [aa_encoder, Lambda(lambda x:x.transpose(1, 2))
                 #,PrintLayer0() 
                 ,length_regulator
                 #,PrintLayer1()
        ]

        last_chans = aa_encoder.embedding_dim
        for i in range(blocks):
            block, new_chans = block_template.build_block(i, last_chans)

            if skips_blocks:
                layers.append(SkipConnection(
                    block,
                    nn.Conv1d(last_chans, new_chans, 1) if last_chans != new_chans else None
                ))
            else:
                layers.append(block)

            last_chans = new_chans
            
        layers.extend([
            Lambda(lambda x:x.mean(dim=2)),  # global average pooling
            nn.Linear(in_features=last_chans, out_features=self._out_dim),
            nn.Dropout(dropout),
        ])
        
        super().__init__(*layers)

    def output_shape(self):
        return self._out_dim
      

class PredictorHead(nn.Sequential):
    def __init__(self, input_size, blocks=3, factor=2, dropout=0.5):
        layers = [nn.Dropout(dropout)]
        for i in range(blocks):
            layers.extend([
                nn.Linear(input_size // int(factor**i), input_size // int(factor**(i + 1))),
                nn.BatchNorm1d(input_size // int(factor**(i + 1))),
                nn.LeakyReLU(),
            ])

        self._output_shape = input_size // int(factor**(i + 1))
        super().__init__(*layers)

    def output_shape(self):
        return self._output_shape

class NoiseLayer(nn.Module):

    """
    
    the noise weight, b, is intialized at -4.6 -> sigmoid(-4.6) = 0.01
    after each update it is refilled with -4.6 (in _fit_batch) until self._keep_noise_constant is set to False
    on default (start_noise = None) self._keep_noise_constant gets switched to False after first convergence
    otherwise, start_noise activates the noise_layer after a epoch == start_noise

    for regularization, noise_reg(self) = abs(0.5 - b) is added to the loss
    
    """

    def __init__(self, device):
        super().__init__()
        self.device = device
        self.b = nn.Linear(1,1,bias = False)
        self.b.weight.data.fill_(-4.6)
 
    def forward(self, noisy_outs):
        clean_outs =  noisy_outs + torch.sigmoid(self.b.weight) - torch.sigmoid(self.b.weight) * noisy_outs
        clean_outs =  clean_outs.view(len(noisy_outs))
        return clean_outs

    def noise_reg(self):
        noise_reg = torch.sigmoid(self.b.weight)
        noise_reg = abs(0.5 - noise_reg)
        return noise_reg.view([])

class CleavagePredictorNetwork(nn.Module):
    def __init__(self, length_regulator, protein_encoder, attention_model, head, noise_layer, device, batch_size):
        super().__init__()
        self.length_regulator = length_regulator
        self.protein_model = protein_encoder
        self.attention_model = attention_model
        self.head = head
        self.noise_layer = noise_layer
        self.output = nn.Linear(self.head.output_shape(), 1)
        self.loss_fun = nn.BCEWithLogitsLoss()
        self.device = device
        #self.noise_layer.b.requires_grad = False
       
    def forward(self, mask, proteins, label=None):  
        xprot = self.protein_model(proteins)
        if torch.sum(torch.isnan(xprot))>0:
           #import pdb; pdb.set_trace()
           print('xprot are nan', xprot)
        
        alphas, att_weights = self.attention_model(xprot, mask)
        if torch.sum(torch.isnan(att_weights))>0:
           print('att_weights are nan', att_weights, 'xprot', xprot, 'proteins', proteins)
        x = self.head(att_weights)
        if torch.sum(torch.isnan(x))>0:
           print('x is nan', x)
        # compute noisy predictions
        noisy_outs = self.output(x).squeeze()
        if torch.sum(torch.isnan(noisy_outs))>0:
           #import pdb; pdb.set_trace()
           print('noisy_outs are nan', noisy_outs)
        # apply noise layer
        #outs = noisy_outs
        outs = self.noise_layer(noisy_outs)
        # compute loss if necessary
        loss = None
        if torch.sum(torch.isnan(outs))>0:
           #import pdb; pdb.set_trace()
           print('outs are nan', outs)
        if torch.sum(torch.isnan(label))>0:
           #import pdb; pdb.set_trace()
           print('label are nan', label)
        if label is not None:
            loss = self.loss_fun(outs, label)
            loss += 0.5e-0 * self.length_regulator.length_reg()
            loss += self.noise_layer.noise_reg()
            return Predictions(logits = outs,alphas = alphas,loss = loss)
        else:         
            return Predictions(logits = outs,alphas = alphas)


                



class AttentionModule(nn.Module, abc.ABC):

    @abc.abstractmethod
    def output_shape(self):
        pass

    @abc.abstractmethod
    def forward(self, xprot, mask):
        '''
        compute attention between peptides and mhcs. return a tuple with the
        attention weights and the weighted results of shape (bags, d)
        xprot.shape = (instances, dim_prot)
        mask.shape = (bags, instances)
        '''


class KeyedAttention(AttentionModule):

    def __init__(self, protein_size, size=128, separate_key_value=True, batch_size=512):
        super().__init__()
        self._size = size
        self._separate_key_value = separate_key_value
        self._batch_size = batch_size
        self.key_network = nn.Linear(protein_size, self._size)
        self.query_network = nn.Linear(protein_size, self._size)
        self.bnk = nn.BatchNorm1d(self._size)
        self.bnq = nn.BatchNorm1d(self._size)
        if self._separate_key_value:
           self.value_network = nn.Linear(protein_size, self._size)
           self.bnv = nn.BatchNorm1d(self._size)
        else:
           self.value_network = self.key_network
           self.bnv = self.bnk 

    def output_shape(self):
        return self._size

    def forward(self, xprot, mask):
        # proteins is a matrix of 30-mers, one per row
        # mask is a binary matrix telling which 30-mer (on columns)
        # goes with which bag (on rows)
        keys = self.bnk(self.key_network(xprot))  # shape: (prots, _size)
        queries = self.bnq(self.query_network(xprot))
        values = self.bnk(self.key_network(xprot))
        weights = keys @ queries.t()/(self._batch_size)  #shape: (prots,prots)
        if torch.sum(torch.isnan(weights)) > 0:
          print('weights are nan', weights, queries, keys)
        exps = torch.exp(weights)
        if torch.sum(torch.isnan(exps)) > 0:
          print('exps are nan', exps, 'prots',keys, queries,'weights', weights, 'mean weights', torch.mean(weights), 'mean prots', torch.mean(prots), 1/np.sqrt(self._size) )

        mask = mask.unsqueeze(0).expand(1,-1, -1)
        mask = mask.view(mask.shape[-2],mask.shape[-1])
        prot_mask = mask + (1-mask)*1e-6 + mask*1e-6 #shape: (bag_size, prots)   
        if torch.sum(torch.isnan(prot_mask)) > 0:
          print('prot_mask are nan', prot_mask)

        mask = prot_mask.t() @ prot_mask
        mask =  mask + (1-mask)*1e-6 + mask*1e-6 #shape: (prots,prots)
        if torch.sum(torch.isnan(mask)) > 0:
          print('mask are nan', mask)

        masked_exps = exps * mask #shape(prots,prots)
        if torch.sum(torch.isnan(masked_exps)) > 0:
          print('masked_exps are nan', masked_exps)
        bag_normalizer = torch.sum(masked_exps @ prot_mask.t(), dim=0) #shape: (bag_size)
        if torch.sum(torch.isnan(bag_normalizer)) > 0:
          print('bag_normalizer are nan', bag_normalizer)
        expanded_normalizer = mask * torch.unsqueeze(bag_normalizer @ prot_mask, 0)+1e-6 #shape: (prots,prots)
        if torch.sum(torch.isnan(expanded_normalizer)) > 0:
          print('expanded_normalizer are nan', expanded_normalizer)

        prot_weights = masked_exps.sum(dim=0) / (1e-6+(bag_normalizer @ prot_mask)) #shape: (prots)
        if torch.sum(torch.isnan(prot_weights)) > 0:
          print('prot_weights are nan', prot_weights)

        bag_weighted_prot_values = prot_mask @ (values.t() * prot_weights).t() #shape: (bag_size,_size)
        if torch.sum(torch.isnan(bag_weighted_prot_values)) > 0:
          print('exps are nan', 'prots shape', keys.shape,'max weights',torch.max(torch.abs(weights)),exps, 'prots',keys,queries,values,'weights', weights, 'mean weights', torch.mean(weights), 'mean prots', torch.mean(keys),'max prots',torch.max(keys), torch.max(queries), 1/np.sqrt(self._size) )
          print('bag_weighted_prot_values are nan', bag_weighted_prot_values, 'pw', prot_weights, 'en', expanded_normalizer, 'bn', bag_normalizer, 'me', masked_exps, 'exps', exps, 'weights', weights, 'prot_mask', prot_mask)
        return masked_exps / expanded_normalizer, bag_weighted_prot_values

class GatedAttention(AttentionModule):
    '''
    gated additive attention where the hidden states are the concatenation
    of mhc and peptide representations.
    D. Bahdanau, K. Cho, and Y. Bengio. “Neural Machine Translation by Jointly
    Learning to Align and Translate”. In: arXiv e-prints abs/1409.0473 (2014).
    '''

    def __init__(self, protein_size, size=128):
        super().__init__()
        self.attention_size = size
        self.output_size = protein_size

        self.u_1 = nn.Linear(protein_size, size)
        self.u_2 = nn.Linear(protein_size, size)

        self.v_1 = nn.Linear(protein_size, size)
        self.v_2 = nn.Linear(protein_size, size)

        self.w_1 = nn.Linear(size, 1)
        self.w_2 = nn.Linear(size, 1)

    def output_shape(self):
        return self.output_size

    def forward(self, xprot, mask):
        # xpep.shape = (pep_instances, dim_pep)
        # peptide_mask.shape = (bags, pep_instances)
        # xmhc.shape = (mhc_instances, dim_mhc)
        # mask.shape = (bags, mhc_instances)

        # compute weights separately for each peptide/mhc
        # shapes: (pep_instances, 1) and (mhc_instances, 1)
        prots1 = self.w_1(torch.tanh(self.v_1(xprot)) * torch.sigmoid(self.u_1(xprot)))
        prots2 = self.w_2(torch.tanh(self.v_2(xprot)) * torch.sigmoid(self.u_2(xprot)))
        weights = prots1 + prots2.t()

        exps = torch.exp(weights)
        if torch.sum(torch.isnan(exps)) > 0:
          print('exps are nan', exps, 'prots',prots1, ports2,'weights', weights, 'mean weights', torch.mean(weights), 'mean prots', torch.mean(prots), 1/np.sqrt(self._size) )

        prot_mask = mask + (1-mask)*1e-6 + mask*1e-6 #shape: (bag_size, prots)   
        if torch.sum(torch.isnan(prot_mask)) > 0:
          print('prot_mask are nan', prot_mask)

        mask = prot_mask.t() @ prot_mask
        mask =  mask + (1-mask)*1e-6 + mask*1e-6 #shape: (prots,prots)
        if torch.sum(torch.isnan(mask)) > 0:
          print('mask are nan', mask)

        masked_exps = exps * mask #shape(prots,prots)
        if torch.sum(torch.isnan(masked_exps)) > 0:
          print('masked_exps are nan', masked_exps)
        bag_normalizer = torch.sum(masked_exps @ prot_mask.t(), dim=0) #shape: (bag_size)
        if torch.sum(torch.isnan(bag_normalizer)) > 0:
          print('bag_normalizer are nan', bag_normalizer)
        expanded_normalizer = mask * torch.unsqueeze(bag_normalizer @ prot_mask, 0)+1e-6 #shape: (prots,prots)
        if torch.sum(torch.isnan(expanded_normalizer)) > 0:
          print('expanded_normalizer are nan', expanded_normalizer)

        prot_weights = masked_exps.sum(dim=0) / (1e-6+(bag_normalizer @ prot_mask)) #shape: (prots)
        if torch.sum(torch.isnan(prot_weights)) > 0:
          print('prot_weights are nan', prot_weights)

        bag_weighted_prot_values = prot_mask @ (xprot.t() * prot_weights).t() #shape: (bag_size,_size)
        if torch.sum(torch.isnan(bag_weighted_prot_values)) > 0:
          print('exps are nan', 'prots shape', prots1.shape,'max weights',torch.max(torch.abs(weights)),exps, 'prots',prots1,prots2,'weights', weights, 'mean weights', torch.mean(weights), 'mean prots', torch.mean(prots1),'max prots',torch.max(prots1), torch.max(prots2), 1/np.sqrt(self._size) )
          print('bag_weighted_prot_values are nan', bag_weighted_prot_values, 'pw', prot_weights, 'en', expanded_normalizer, 'bn', bag_normalizer, 'me', masked_exps, 'exps', exps, 'weights', weights, 'prot_mask', prot_mask)
        return masked_exps / expanded_normalizer, bag_weighted_prot_values

