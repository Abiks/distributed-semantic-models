from functools import partial

import torch
from torch.utils.data import Dataset, DataLoader
from allennlp.modules.elmo import batch_to_ids

from .udpipe_preprocessing import preprocess

class NERDataset(Dataset):
    def __init__(self, corpus):
        self.samples = []
        for sample in corpus:
            self.samples.append(
                ([s[0] for s in sample], [s[1] for s in sample])
            )
        self.corpus_len = len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return self.corpus_len


def make_batch_raw(batch, labels_vocab):
    initial_texts = [sample[0] for sample in batch]
    texts = [preprocess(' '.join(txt)) for txt in initial_texts]
    
    initial_labels = [sample[1] for sample in batch]
    forward_ids = batch_to_ids(texts)
    
    max_len = max(len(e) for e in initial_labels)
    mask = []
    padded_labels = []
    
    for labels in initial_labels:
        length = len(labels)
        mask.append([1] * length + [0] * (max_len - length))
        padded_labels.append([labels_vocab[lbl] for lbl in labels] + [0] * (max_len - length))
    mask = torch.tensor(mask)
    padded_labels = torch.tensor(padded_labels)

    return {
        'ids': forward_ids,
        'mask': mask,
        'backward_ids': None,
        'target': padded_labels,
        'initial_texts': initial_texts
    }


    

def get_datasets(path_to_data):
    flist = ['test.conllu', 'train.conllu', 'val.conllu']
    for fname in flist:
        corpus = []
        with (path_to_data / fname).open(encoding='utf-8') as f:
            line = f.readline()
            buffer = []
            while line:
                line = line.strip()
                if line:
                    line = line.split()
                    if line[1].isalnum():
                        buffer.append((line[1].lower(), line[2]))
                else:
                    if buffer:
                        corpus.append(buffer)
                    buffer = []
                line = f.readline()
        
        if 'test' in fname:
            test_dataset = NERDataset(corpus)
        elif 'val' in fname:
            val_dataset = NERDataset(corpus)
        elif 'train' in fname:
            train_dataset = NERDataset(corpus)
    return train_dataset, val_dataset, test_dataset


def get_dataloaders(ner_task, batch_size, **kwargs):
    path_to_data = ner_task['path']
    train_dataset, val_dataset, test_dataset = get_datasets(path_to_data)
    make_batch = partial(make_batch_raw, labels_vocab=ner_task['labels_vocab'])

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size,
        collate_fn=make_batch, 
        **kwargs,
    )

    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size,
        collate_fn=make_batch,
        **kwargs,
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size,
        collate_fn=make_batch,
        **kwargs,
    )

    return train_dataloader, val_dataloader, test_dataloader


