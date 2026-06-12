import random
import numpy as np
import torch

from datasets import load_dataset, load_from_disk
from torch.utils.data.dataset import Dataset

from tqdm import tqdm

# get data online
# def get_c4(tokenizer, n_samples, seq_len):
#     traindata = load_dataset(
#         'allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
#     )
    
#     tokenized_samples, history = [], []
#     for _ in range(n_samples):
#         while True:
#             i = random.randint(0, len(traindata) - 1)
#             tokenized_sample = tokenizer(traindata[i]['text'], return_tensors='pt')
#             if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
#                 history.append(i)
#                 break
#         i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len )
#         tokenized_samples.append(tokenized_sample.input_ids[:, i:i+seq_len])
#     return torch.cat(tokenized_samples, dim=0)

# def get_bookcorpus(tokenizer, n_samples, seq_len):
#     # traindata = load_dataset(
#     #     'bookcorpus', split='train'
#     # )
#     traindata = load_dataset(
#         'Yuti/bookcorpus', split='train'
#     )
    
#     tokenized_samples, history = [], []
#     for _ in range(n_samples):
#         while True:
#             i = random.randint(0, len(traindata) - 1)
#             tokenized_sample = tokenizer(traindata[i]['text'], return_tensors='pt')
#             if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
#                 history.append(i)
#                 break
#         i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)
#         tokenized_samples.append(tokenized_sample.input_ids[:, i:i+seq_len])
#     return torch.cat(tokenized_samples, dim=0 )

# get data offline
def get_c4(tokenizer, n_samples, seq_len):
    traindata = load_from_disk("./offline_datasets/c4_en_5000")
    
    tokenized_samples, history = [], []
    for _ in tqdm(range(n_samples)):
        while True:
            i = random.randint(0, len(traindata) - 1)
            tokenized_sample = tokenizer(traindata[i]["text"], return_tensors="pt")
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break

        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i+seq_len])

    return torch.cat(tokenized_samples, dim=0)


def get_gsm8k(tokenizer, n_samples, seq_len):
    traindata = load_dataset("/mnt/foxbrain-omni-distillation/academia_sinica/distill/datasets/openai/gsm8k", "main")
    traindata = traindata["train"]

    def build_text(examples):
        texts = [
            f"{q}\n{a}" 
            for q, a in zip(examples["question"], examples["answer"])
        ]
        return {"text": texts}
        
    traindata = traindata.map(build_text, batched=True)

    tokenized_samples, history = [], []
    for _ in tqdm(range(n_samples)):
        while True:
            i = random.randint(0, len(traindata) - 1)
            tokenized_sample = tokenizer(traindata[i]["text"], return_tensors="pt")
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break

        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i+seq_len])

    return torch.cat(tokenized_samples, dim=0)


def get_bookcorpus(tokenizer, n_samples, seq_len):
    traindata = load_from_disk("/mnt/foxbrain-omni-distillation/academia_sinica/pruning/datasets/bookcorpus_yuti_5000")
    
    tokenized_samples, history = [], []
    for _ in tqdm(range(n_samples)):
        while True:
            i = random.randint(0, len(traindata) - 1)
            tokenized_sample = tokenizer(traindata[i]["text"], return_tensors="pt")
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break

        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i+seq_len])

    return torch.cat(tokenized_samples, dim=0)

def get_examples(dataset, tokenizer, n_samples, seq_len = 128):
    if '+' in dataset:
        dataset_names = dataset.split('+')

        if len(dataset_names) > 2:
            raise ValueError("This function currently supports mixing a maximum of 2 datasets.")
    
        half_samples = n_samples // 2
        remaining_samples = n_samples - half_samples
        
        part1 = get_examples(dataset_names[0], tokenizer, remaining_samples, seq_len)
        part2 = get_examples(dataset_names[1], tokenizer, half_samples, seq_len)
        
        combined_tensor = torch.cat([part1, part2], dim=0)
        shuffle_indices = torch.randperm(combined_tensor.size(0))
        
        return combined_tensor[shuffle_indices]
        
    if dataset == 'c4':
        return get_c4(tokenizer, n_samples, seq_len)
    elif dataset == 'bookcorpus':
        return get_bookcorpus(tokenizer, n_samples, seq_len)
    elif dataset == 'gsm8k':
        return get_gsm8k(tokenizer, n_samples, seq_len)
    else:
        raise NotImplementedError
