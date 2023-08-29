from transformers import AutoTokenizer

import pandas as pd
from datasets import DatasetDict, Dataset, load_dataset
import torch


model_checkpoint = "vinai/vinai-translate-en2vi"
# model_checkpoint = "VietAI/envit5-translation"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, src_lang = 'en_XX', tgt_lang='vi_VN')

tokenizer.save_pretrained("./models/tokenizer_vinai/")
# tokenizer.save_pretrained("./models/tokenizer_vit5/")