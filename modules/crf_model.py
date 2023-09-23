from collections import Counter

import numpy as np
import pytorch_lightning as pl
import torch
from torchcrf import CRF
from sklearn.metrics import f1_score


def aggregate_word_prediction(w_preds):
    return Counter(w_preds).most_common(1)[0][0]


class CRFModel(pl.LightningModule):
    def __init__(self, ner_labels, vectorizer, weight_decay, learning_rate=1e-3):
        super(CRFModel, self).__init__()
        self.vectorizer = vectorizer
        emb_size = vectorizer.emb_size

        self.lstm = torch.nn.LSTM(emb_size, 120, num_layers=2, batch_first=True, bidirectional=True)
        self.relu = torch.nn.ReLU()
        self.line = torch.nn.Linear(240, len(ner_labels))
        self.crf = CRF(num_tags=len(ner_labels), batch_first=True)
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.ner_labels = ner_labels
        self.history = []
        self.validation_step_preds = []
        self.validation_step_gts = []
                
    def get_tail_norm(self):
        l2 = 0
        with torch.no_grad():
            for layer in [self.lstm, self.line, self.crf]:
                for p in layer.parameters():
                    l2 += (p ** 2).sum().item()
        l2 = np.sqrt(l2)
        return l2        

    def training_step(self, batch):
        self.vectorizer.eval()
        embeddings = self.vectorizer.vectorize(batch)

        transformed = embeddings
        if isinstance(transformed, tuple):
            transformed = transformed[0]
        transformed, _ = self.lstm(transformed)
        transformed = self.relu(transformed)
        transformed = self.line(transformed)
        mask = batch['mask'].bool()
        return self.crf(transformed, batch['target'], mask) * (-1) / batch['mask'].shape[0]
    
    def validation_step(self, batch, batch_idx):
        embeddings = self.vectorizer.vectorize(batch)
        transformed = embeddings
        if isinstance(transformed, tuple):
            transformed = transformed[0]
        transformed, _ = self.lstm(transformed)
        transformed = self.relu(transformed)
        transformed = self.line(transformed)
        mask = batch['mask'].bool()
        predictions = self.crf.decode(transformed, mask)
        lens = batch['mask'].sum(1)
        targets = [batch['target'][i, :lens[i]] if lens[i] < batch['target'].shape[1] else batch['target'][i] for i in range(batch['mask'].shape[0])]
        targets = [t.tolist() for t in targets]
        for p in predictions:
            self.validation_step_preds.extend(p)
        for t in targets:
            self.validation_step_gts.extend(t)
    
    def on_validation_epoch_end(self):
        f1 = f1_score(y_true=self.validation_step_gts, y_pred=self.validation_step_preds,
            average='weighted', labels=np.arange(1, len(self.ner_labels)), zero_division=0)
        self.log("f1_tokens", f1)
        self.history.append(f1)
        self.validation_step_preds.clear()
        self.validation_step_gts.clear()

    def predict_step(self, batch, batch_idx):
        word_predictions = []
        flatten_target = []
        predictions, _ = self.validation_step(batch, 0)
        for word_indices, preds in zip(batch['batch_word_indices'], predictions):
            if len(word_indices) == 2:
                continue
            assert len(word_indices) == len(preds)

            buffer = []
            current_word_index = 0
            for w_idx, pred in zip(word_indices, preds):
                if w_idx == -1:
                    continue

                if w_idx != current_word_index:
                    word_predictions.append(
                        aggregate_word_prediction(buffer)
                    )
                    buffer = [pred]
                    current_word_index = w_idx
                else:
                    buffer.append(pred)

            word_predictions.append(
                aggregate_word_prediction(buffer)
            )

        for t in batch['initial_labels']:
            flatten_target.extend([self.ner_labels[lbl] for lbl in t])
        return word_predictions, flatten_target
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer
