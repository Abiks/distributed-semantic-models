from torch.utils.data import IterableDataset, DataLoader
import re
import torch
from allennlp.modules.elmo import batch_to_ids


class CustomIterableDataset(IterableDataset):
    def __init__(self, filename, length=None):
        self.filename = filename
        if length is None:
            self.length = int(re.search('\d+', filename).group(0))
        else:
            self.length = length

    def __iter__(self):
        file_itr = open(self.filename)
        return file_itr
    
    def __len__(self):
        return self.length


def make_batch(batch, token_dict):
    tokenized_texts = []
    max_len = 0 
    for text in batch:
        tokenized_text = text.strip().split()        
        tokenized_texts.append(tokenized_text)
        max_len = max(max_len, len(tokenized_text))
    ids = batch_to_ids(tokenized_texts)
    
    forward_target = []
    backward_target = []
    for tokenized_text in tokenized_texts:
        token_ids = [token_dict.get(token, token_dict['<UNK>']) for token in tokenized_text]
        forward_target.append(
            token_ids[1:] + [token_dict['</S>']] + [token_dict['<PAD>']] * (max_len - len(tokenized_text))
        )
        backward_target.append(
            [token_dict['<S>']] + token_ids[:-1] + [token_dict['<PAD>']] * (max_len - len(tokenized_text))
        )

    return {
        'ids': ids,
        'forward_target': torch.LongTensor(forward_target),
        'backward_target': torch.LongTensor(backward_target)
    }

def load_token_dict(path_to_vocab):
    token_dict = {'<PAD>':0}
    idx = 1
    vocab = []
    with open(path_to_vocab) as f:
        line = f.readline()
        while line:
            token_dict[line.strip()] = idx
            vocab.append(line.strip())
            idx += 1
            line = f.readline()
    return token_dict
    