1. Run `read_dataset.py` to upload the dataset to HuggingFace Hub.
2. Run `get_tokenize.py` to save tokenizers locally.
3. Set up environment variables `setx WANDB_PROJECT en2vi`.
4. Run `CUDA_VISIBLE_DEVICES=0,1,2 python finetuning_vinai.py` to track the training process.

To change when vi2en:
1. Fix in file `get_tokenize.py` to get new tokenizer
1. Swap in line 30.

