from pathlib import Path


RUSSIAN = {
    'MultiCoNER': {
        'path': Path('/media/fast_data/ELMO-SRU/MultiCoNER'),
        'label_list': ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-CORP', 'I-CORP', 'B-GRP', 'I-GRP', 'B-PROD', 'I-PROD', 'B-CW', 'I-CW']
    },
    'WikiNeural': {
        'path': Path('/media/fast_data/ELMO-SRU/WikiNeural'),
        'label_list': ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC'],
    }
}


for key in RUSSIAN:
    RUSSIAN[key]['labels_vocab'] = {v:idx for idx, v in enumerate(RUSSIAN[key]['label_list'])}
