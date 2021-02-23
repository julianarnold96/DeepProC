
from train import TrainingConfigurator


class Config(TrainingConfigurator):
   def __init__(self):
      self.cv_do: bool = False
      self.cv_repetitions: int = 2
      self.cv_folds: int = 10
      self.cv_seed = 945271
      self.batch_size = 512
      self.max_epochs = None
      self.output_dir = '/home/icb/julian.arnold'
      self.dataset_path = '/content/drive/My Drive/mhc1_data.csv'
      self.sample: int = None
      self._device: str = 'auto'
      self.negatives_generator = 'shifting_window_negatives'
      self.number_of_negatives: int = 1
      self.nterm: int = 0 # if set to 1, n-terminal cleavage sites are also used 
      self.punishment_weight_left = 0.112
      self.punishment_weight_right = 0.525
      self.aa_left: int = 15
      self.aa_right: int = 15
      self.mhc: int = 1
      self.use_split = False
      self.start_noise = 1 # sets epoch after which noise layer gets updated (0 activates after first epoch, None after first convergence)

   def get_predictor(self):
        from model import CleavagePredictor

        net = self._network()
        opt = self._optimizer(net.parameters())
        return CleavagePredictor(net, opt)


   def _optimizer(self, params):
        from torch import optim
        return optim.Adam(params, lr=1e-3)

   def _network(self):
        from torch import nn
        from model import CleavagePredictorNetwork

        aa_encoder = self._get_aa_encoder()
        length_regulator = self._get_length_regulator()   
        pep_enc = self._get_peptide_encoder(aa_encoder, length_regulator)
        attn = self._get_attention(pep_enc.output_shape())
        head = self._get_predictor_head(attn.output_shape())
        noise_layer = self._get_noise_layer()

        nnet = CleavagePredictorNetwork(length_regulator, pep_enc, attn, head, noise_layer, self.device, self.batch_size)
        nnet.to(self.device)

        return nnet

   def _get_aa_encoder(self):
        from torch.nn import Embedding
        from dataset import RandomEncoder, Blosum50Encoder, OneHotEncoder

        embs = Blosum50Encoder().get_weights_as_torch_tensor().to(self.device)

        aa_encoder = Embedding(embs.shape[0], embs.shape[1], padding_idx=-1, _weight=embs)
        return aa_encoder

   def _get_length_regulator(self):
        from model import PositionalLengthRegularization
        length_regulator = PositionalLengthRegularization(self.device, punishment_weight_left = self.punishment_weight_left, punishment_weight_right = self.punishment_weight_right, aa_left = self.aa_left, aa_right = self.aa_right) 
        return length_regulator

   def _get_peptide_encoder(self, aa_encoder, length_regulator):
        from model import PeptideEncoder, VggBlock, InceptionBlock

        block = VggBlock(
            base=128,
            factor=1.5,
            count=5,
        )

        pep_enc = PeptideEncoder(
            aa_encoder,
            self.device,
            length_regulator, 
            block_template=block,
            blocks=2,
            out_dim=32,
            dropout=0.0,
            skips_blocks=True
        )

        return pep_enc

   def _get_attention(self, pep_size):
        from model import GatedAttention, KeyedAttention

        attn = KeyedAttention(
            pep_size,
            size=128,
            separate_key_value=True,
        )

        return attn

   def _get_noise_layer(self):
        from model import NoiseLayer
        
        noise_layer = NoiseLayer(self.device) 
         
        return noise_layer

   def _get_predictor_head(self, attention_size):
        from model import PredictorHead

        head = PredictorHead(
            attention_size,
            blocks=2,
            factor=1.5,
            dropout=0.1,
        )

        return head

configurator = Config()
