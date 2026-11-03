"""Exercise 2: seeded, auditable train/validation/test preparation."""
from collections import Counter
from datasets import load_dataset
from transformers import AutoTokenizer
from config import (DATASET, DATA_DIR, FINETUNE_MODEL, MAX_LENGTH, SEED,
                    TRAIN_SAMPLES, VALIDATION_SAMPLES, TEST_SAMPLES)
from utils import sha256_records, write_json

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_dataset(DATASET)
    assert {'train', 'test'} <= set(raw), 'Expected official train and test splits'
    assert len(raw['train']) >= TRAIN_SAMPLES + VALIDATION_SAMPLES
    assert len(raw['test']) >= TEST_SAMPLES
    pool = raw['train'].shuffle(seed=SEED).select(range(TRAIN_SAMPLES + VALIDATION_SAMPLES))
    parts = pool.train_test_split(test_size=VALIDATION_SAMPLES, seed=SEED,
                                  stratify_by_column='label')
    original = {
        'train': parts['train'], 'validation': parts['test'],
        'test': raw['test'].shuffle(seed=SEED).select(range(TEST_SAMPLES))
    }
    # Reject any identical review across splits (test labels remain untouched until final evaluation).
    sets = {k: set(str(t).strip() for t in ds['text']) for k, ds in original.items()}
    for a,b in [('train','validation'), ('train','test'), ('validation','test')]:
        assert not (sets[a] & sets[b]), f'Text overlap found between {a} and {b}'
    tokenizer = AutoTokenizer.from_pretrained(FINETUNE_MODEL)
    def tokenize(batch):
        return tokenizer(batch['text'], truncation=True, max_length=MAX_LENGTH)
    meta = {'dataset_id': DATASET, 'model_id': FINETUNE_MODEL,
            'seed': SEED, 'max_length': MAX_LENGTH, 'label_map': {'0':'NEGATIVE','1':'POSITIVE'},
            'splits': {}, 'selection_protocol':
              'Train and select by validation only. Final test once after configuration is frozen.'}
    for name, ds in original.items():
        labels = Counter(int(x) for x in ds['label'])
        meta['splits'][name] = {'count':len(ds), 'labels': {'NEGATIVE':labels[0], 'POSITIVE':labels[1]},
                                'text_label_sha256':sha256_records(ds)}
        prepared = ds.map(tokenize, batched=True, remove_columns=['text'])
        prepared.save_to_disk(str(DATA_DIR / name))
        print(name, len(prepared), dict(labels))
    write_json(DATA_DIR / 'split_info.json', meta)
    print('Data prepared:', DATA_DIR / 'split_info.json')

if __name__ == '__main__':
    main()
