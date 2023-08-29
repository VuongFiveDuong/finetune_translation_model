from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq

import pandas as pd
from datasets import DatasetDict, Dataset, load_dataset
import torch
import logging
import sys


# Setup logging
logging.basicConfig(
  level = logging.INFO,
  format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
  datefmt="%m/%d/%Y %H:%M:%S",
  filename='finetuning.log', 
  encoding='utf-8'
)
logger = logging.getLogger(__name__)

# Change name in line 


# DATA PREPROCESSING
dataset = load_dataset("vtd/EVMed")
tokenizer = AutoTokenizer.from_pretrained("./models/tokenizer_vinai", src_lang = "en_XX", tgt_lang = "vi_VN")

prefix = ""
max_length = 512

def preprocess_function(examples):
    inputs = [prefix + example["en"] for example in examples["translation"]]
    targets = [example["vi"] for example in examples["translation"]]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=max_length, truncation=True)
    return model_inputs

train_dataset = dataset["train"]
train_dataset = train_dataset.select(range(3500))
train_dataset = train_dataset.map(
    preprocess_function,
    batched=True
)

val_dataset = dataset["val"]
val_dataset = val_dataset.select(range(100))
val_dataset = val_dataset.map(
    preprocess_function,
    batched=True
)

test_dataset = dataset["test"]
test_dataset = test_dataset.select(range(100))
test_dataset = test_dataset.map(
    preprocess_function,
    batched=True
)


# # MODEL PREPARATION 

model = AutoModelForSeq2SeqLM.from_pretrained("vinai/vinai-translate-en2vi", # Change to checkpoint if resume from checkpoint
                                              decoder_start_token_id=tokenizer.lang_code_to_id["vi_VN"]
)

# If train new model from scratch
# from transformers import AutoConfig


# config = AutoConfig.from_pretrained("vinai/vinai-translate-en2vi", 
#                                     decoder_start_token_id=tokenizer.lang_code_to_id["vi_VN"])
# model = AutoModelForSeq2SeqLM.from_config(config)



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
    output_dir="./result_vinai_test",
    evaluation_strategy="steps",
    eval_steps=100,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,  
    weight_decay=0.01,
    save_total_limit=1,
    load_best_model_at_end=True,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=True,
    report_to="wandb",
    run_name="your_favourite_name"    
)

logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
logger.info(f"Training/evaluation parameters {training_args}")

# model = AutoModelForSeq2SeqLM.from_pretrained("vinai/vinai-translate-en2vi")

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

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
print([ex['vi'] for ex in test_dataset["translation"]])
print([ex['en'] for ex in test_dataset["translation"]])
exit(0)


metrics = predict_results.metrics
max_predict_samples = (
    data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
)
metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

trainer.log_metrics("predict", metrics)
trainer.save_metrics("predict", metrics)

if trainer.is_world_process_zero():
    if training_args.predict_with_generate:
        predictions = predict_results.predictions
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        predictions = tokenizer.batch_decode(
            predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        predictions = [pred.strip() for pred in predictions]
        output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            writer.write("\n".join(predictions))



