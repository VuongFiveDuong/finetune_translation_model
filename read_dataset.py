from transformers import AutoTokenizer

import pandas as pd
from datasets import DatasetDict, Dataset
import torch


# DATA PREPROCESSING
# Load training set
with open('data/EVMed/train/train.en.txt', 'r', encoding="utf8") as f:
    train_en = f.readlines()
with open('data/EVMed/train/train.vi.txt', 'r', encoding="utf8") as f:
    train_vi = f.readlines()
training_set = [{'en': en_sentence, 'vi': vi_sentence} for en_sentence, vi_sentence in zip(train_en, train_vi)]

# Load validation set
with open('data/EVMed/val/val.en.txt', 'r', encoding="utf8") as f:
    val_en = f.readlines()
with open('data/EVMed/val/val.vi.txt', 'r', encoding="utf8") as f:
    val_vi = f.readlines()
validation_set = [{'en': en_sentence, 'vi': vi_sentence} for en_sentence, vi_sentence in zip(val_en, val_vi)]

# Load test set
with open('data/EVMed/test/test.en.txt', 'r', encoding="utf8") as f:
    test_en = f.readlines()
with open('data/EVMed/test/test.vi.txt', 'r', encoding="utf8") as f:
    test_vi = f.readlines()
test_set = [{'en': en_sentence, 'vi': vi_sentence} for en_sentence, vi_sentence in zip(test_en, test_vi)]

dataset = DatasetDict({
    'train': Dataset.from_dict({'translation': training_set}),
    'val': Dataset.from_dict({'translation': validation_set}),
    'test': Dataset.from_dict({'translation': test_set}),
})

dataset.push_to_hub("vtd/EVMEd", private=True)