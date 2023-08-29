from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq

import pandas as pd
from datasets import DatasetDict, Dataset, load_dataset
import torch
import logging
import sys
import argparse
 
parser = argparse.ArgumentParser(description="What to do with this file?",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-f", "--finetune", default=True,
                    help="Finetune or train from scratch")
args = parser.parse_args()

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    filename='finetuning.log', 
    encoding='utf-8'
)
logger = logging.getLogger(__name__)


# DATA PREPROCESSING
dataset = load_dataset("vtd/EVMed")
tokenizer = AutoTokenizer.from_pretrained("./models/tokenizer_vit5_en2vi", src_lang = "en_XX", tgt_lang = "vi_VN")  # CHANGE HERE

src_lang = "en"  # CHANGE HERE
tgt_lang = "vi"  # CHANGE HERE

prefix = ""
max_length = 512

def preprocess_function(examples):
    inputs = [prefix + src_lang + ": " + example[src_lang] for example in examples["translation"]]
    targets = [tgt_lang + ": " + example[tgt_lang] for example in examples["translation"]]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=max_length, truncation=True)
    return model_inputs

train_dataset = dataset["train"]
train_dataset = train_dataset.select(range(350))
train_dataset = train_dataset.map(
    preprocess_function,
    batched=True
)

val_dataset = dataset["val"]
val_dataset = val_dataset.select(range(10))
val_dataset = val_dataset.map(
    preprocess_function,
    batched=True
)

test_dataset = dataset["test"]
test_dataset = test_dataset.select(range(10))
test_dataset = test_dataset.map(
    preprocess_function,
    batched=True
)


# # MODEL PREPARATION 
from transformers import AutoConfig


if args.finetune:
    model = AutoModelForSeq2SeqLM.from_pretrained("VietAI/envit5-translation", max_length=max_length)  # Change to checkpoint if resume from checkpoint

# If train new model from scratch
else:
    config = AutoConfig.from_pretrained("VietAI/envit5-translation", max_length=max_length)
    model = AutoModelForSeq2SeqLM.from_config(config)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, 
                                    model=model)

# Metric
import evaluate
import numpy as np


metric = evaluate.load("sacrebleu")

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


# TRAINING
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer


training_args = Seq2SeqTrainingArguments(
    output_dir="./result_vit5_en2vi",
    evaluation_strategy="steps",
    eval_steps=10,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,  
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    predict_with_generate=True,
    # fp16=True
)

# model = AutoModelForSeq2SeqLM.from_pretrained("vinai/vinai-translate-en2vi")

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# print(trainer.evaluate())

logger.info("*** Train ***")

train_result = trainer.train()
# trainer.save_model()  # Saves the tokenizer too for easy upload

metrics = train_result.metrics
metrics["train_samples"] = 3500

trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()
exit(0)

logger.info("*** Predict ***")

predict_results = trainer.predict(
    test_dataset, metric_key_prefix="predict"
)
predictions = predict_results.predictions
predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
predictions = tokenizer.batch_decode(
    predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
)
predictions = [pred.strip() for pred in predictions]
print(predictions)
print([ex[src_lang] for ex in test_dataset["translation"]])
print([ex[tgt_lang] for ex in test_dataset["translation"]])
exit(0)



