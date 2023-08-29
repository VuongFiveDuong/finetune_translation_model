from transformers import AutoTokenizer

import pandas as pd
from datasets import DatasetDict, Dataset, load_dataset
import torch


# model_checkpoint = "vinai/vinai-translate-vi2en"
model_checkpoint = "VietAI/envit5-translation"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, src_lang='vi_VN', tgt_lang='en_XX')

# tokenizer.save_pretrained("./models/tokenizer_vinai_vi2en/")
tokenizer.save_pretrained("./models/tokenizer_vit5_vi2en/")