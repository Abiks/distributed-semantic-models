import pytorch_lightning as pl
import torch

from allennlp.modules.elmo import Elmo
from allennlp.nn.util import remove_sentence_boundaries


class RNNBlockWithProjection(pl.LightningModule):
    def __init__(self, hidden_size, num_layers):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.projection = torch.nn.Linear(
            in_features=hidden_size,
            out_features=512
        )

    def forward(self, x, mask):
        assert torch.isfinite(x).all()
        lstm_out, _ = self.lstm(x)
        if not torch.isfinite(lstm_out).all():
            import pickle 
            with open('wtf.pickle', 'wb') as f:
                pickle.dump((x, self.lstm), f)
                raise AssertionError
        lstm_out = self.projection(lstm_out)
        assert torch.isfinite(lstm_out).all()
        return lstm_out 
  
    
class ElmoSRU(pl.LightningModule):
    def __init__(self, model_params):
        super().__init__()
        self.forward_rnn_block = RNNBlockWithProjection(
            hidden_size=model_params['hidden_size'], 
            num_layers=model_params['num_layers'], 
            ) 
       
        self.backward_rnn_block = RNNBlockWithProjection(           
            hidden_size=model_params['hidden_size'], 
            num_layers=model_params['num_layers'], 
            ) 
        
    def forward(self, x, mask):
        forward_x = x        
        forward_out = self.forward_rnn_block(forward_x, mask)
        assert torch.isfinite(forward_out).all()
        
        backward_x = self.reverse(x, mask)
        backward_out = self.backward_rnn_block(backward_x, mask)
        backward_out = self.reverse(backward_out, mask)
        assert torch.isfinite(backward_out).all()
        
        sru_out = torch.cat((forward_out, backward_out), dim=2)
        assert sru_out.shape[2] == 1024
        return sru_out.unsqueeze(0) 
        
    def reverse(self, x, mask):
        index = torch.LongTensor([
            list(range(s-1, -1, -1)) + list(range(s, x.shape[1])) 
            for s in mask.sum(1)])
        assert index.shape[0] == x.shape[0]
        index = index.to(x.device)
        if len(x.shape) == 3:
            index = index.unsqueeze(2)
        return torch.take_along_dim(x, index, dim=1)


class BidirectionalLM(pl.LightningModule):
    def __init__(self, 
                 model_params, learning_rate, 
                 elmo_options_file, elmo_weight_file, 
                 vocab_size=None, cutoffs=None
                ):
        super().__init__()
        elmo = Elmo(
            options_file=elmo_options_file,
            weight_file=elmo_weight_file,
            num_output_representations=1,
        )
        elmo_sru = ElmoSRU(model_params)
        elmo._elmo_lstm._elmo_lstm = elmo_sru
        self.bilstm = elmo._elmo_lstm
        
        if cutoffs and vocab_size:
            self.forward_loss = torch.nn.AdaptiveLogSoftmaxWithLoss(
                in_features=512, 
                n_classes=vocab_size, 
                cutoffs=cutoffs
            )
            self.backward_loss = torch.nn.AdaptiveLogSoftmaxWithLoss(
                in_features=512, 
                n_classes=vocab_size, 
                cutoffs=cutoffs
            )
        
        self.learning_rate = learning_rate
        self.history = []
        self.prev_smoothed_loss = None
        
    def training_step(self, batch, batch_idx):
        for key in batch:
            assert torch.isfinite(batch[key]).all()
        embeddings, _ = self.vectorize(batch)
        assert torch.isfinite(embeddings).all()
        flatten_embeddings = embeddings.flatten(0, 1)
        
        forward_emb = flatten_embeddings[:,:512]
        backward_emb = flatten_embeddings[:,512:]
        
        forward_loss = self.forward_loss(forward_emb, batch['forward_target'].flatten(0, 1)).loss
        backward_loss = self.backward_loss(backward_emb, batch['backward_target'].flatten(0, 1)).loss
        
        loss = (forward_loss + backward_loss) / 2
        assert torch.isfinite(loss).all()    
        if self.prev_smoothed_loss is None:
            self.prev_smoothed_loss = loss.item()
        else:
            eps = 0.01
            self.prev_smoothed_loss = (1-eps) * self.prev_smoothed_loss + eps * loss.item()
        self.log('loss', self.prev_smoothed_loss, prog_bar=True)
        self.history.append(loss.item())
        return loss
        
    def vectorize(self, batch):
        ids = batch['ids']
        out = self.bilstm(ids)
        assert len(out['activations']) == 2
        embeddings = out['activations'][1]
        mask = out['mask']
        assert embeddings.shape[2] == 1024
        embeddings, mask = remove_sentence_boundaries(embeddings, mask)
        return embeddings, mask
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.learning_rate,
            weight_decay=0.01
        )
        return optimizer
    
