import os
import pickle
import logging
import torch
from datasets import load_dataset
from torch.utils.data import Dataset
import spacy
import random

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

SRC_LANGUAGE = 'en'
TGT_LANGUAGE = 'ja'

# Disable torchtext deprecation warnings
import torchtext
torchtext.disable_torchtext_deprecation_warning()

# Load SpaCy models
nlp_src = spacy.load("en_core_web_sm")
nlp_tgt = spacy.load("ja_core_news_sm")

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

cache_dir = 'cache'
os.makedirs(cache_dir, exist_ok=True)

def save_vocab(vocab, path: str):
    with open(path, 'wb') as f:
        pickle.dump(vocab, f)
    logging.info(f"Vocabulary saved to {path}")

def load_vocab(path: str):
    with open(path, 'rb') as f:
        vocab = pickle.load(f)
    logging.info(f"Vocabulary loaded from {path}")
    return vocab

def load_and_select_dataset(src_lang, tgt_lang):
    logging.info("Loading dataset")
    dataset = load_dataset('opus100', f'{src_lang}-{tgt_lang}')
    logging.info("Selecting samples from dataset")
    selected_samples = dataset['train']
    return selected_samples, dataset['validation']

# Load and select dataset
logging.info("Loading and selecting dataset")
train_data, val_data = load_and_select_dataset(SRC_LANGUAGE, TGT_LANGUAGE)
logging.info(f"Loaded {len(train_data)} training samples and {len(val_data)} validation samples")

def tokenize_spacy(text, lang):
    if lang == SRC_LANGUAGE:
        return [tok.text for tok in nlp_src.tokenizer(text)]
    elif lang == TGT_LANGUAGE:
        return [tok.text for tok in nlp_tgt.tokenizer(text)]
    else:
        raise ValueError(f"Unsupported language: {lang}")

def build_vocab(data, lang):
    tokens = []
    for sample in data:
        tokens.extend(tokenize_spacy(sample['translation'][lang], lang))
    vocab = {token: idx for idx, token in enumerate(set(tokens), len(special_symbols))}
    for i, sym in enumerate(special_symbols):
        vocab[sym] = i
    return vocab

def numericalize(tokens, vocab):
    return [vocab.get(token, UNK_IDX) for token in tokens]

def tensor_transform(token_ids: list[int]):
    return torch.tensor([BOS_IDX] + token_ids + [EOS_IDX])

text_transform = {
    SRC_LANGUAGE: lambda text: tensor_transform(numericalize(tokenize_spacy(text, SRC_LANGUAGE), en_vocab)),
    TGT_LANGUAGE: lambda text: tensor_transform(numericalize(tokenize_spacy(text, TGT_LANGUAGE), ja_vocab))
}

class TranslationDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_text = self.data[idx]['translation'][SRC_LANGUAGE].rstrip("\n")
        tgt_text = self.data[idx]['translation'][TGT_LANGUAGE].rstrip("\n")
        
        # Tokenize and numericalize the raw text
        src_sample = text_transform[SRC_LANGUAGE](src_text)
        tgt_sample = text_transform[TGT_LANGUAGE](tgt_text)
        
        return src_text, tgt_text, src_sample, tgt_sample

# Build vocabularies
logging.info("Building vocabularies")
en_vocab = build_vocab(train_data, SRC_LANGUAGE)
ja_vocab = build_vocab(train_data, TGT_LANGUAGE)

# Save vocabularies
save_vocab(en_vocab, 'en_vocab.pkl')
save_vocab(ja_vocab, 'ja_vocab.pkl')

# Create datasets
logging.info("Creating training and validation datasets")
train_dataset = TranslationDataset(train_data)
val_dataset = TranslationDataset(val_data)

# Sample 1000 random examples
sample_indices = random.sample(range(len(train_dataset)), 1000)
random_samples = [train_dataset[i] for i in sample_indices]

# Print the 1000 random examples of raw, encoded, and decoded sentences
for i, (src_text, tgt_text, src_sample, tgt_sample) in enumerate(random_samples):
    decoded_src = " ".join([k for k, v in en_vocab.items() if v in src_sample])
    decoded_tgt = " ".join([k for k, v in ja_vocab.items() if v in tgt_sample])
    logging.info(f"Example {i+1}:")
    logging.info(f"Raw Src: {src_text}")
    logging.info(f"Encoded Src: {src_sample}")
    logging.info(f"Decoded Src: {decoded_src}")
    logging.info(f"Raw Tgt: {tgt_text}")
    logging.info(f"Encoded Tgt: {tgt_sample}")
    logging.info(f"Decoded Tgt: {decoded_tgt}")

# Save datasets to pickle files
with open('train_dataset.pkl', 'wb') as f:
    pickle.dump(train_dataset, f)
logging.info("Training dataset saved to train_dataset.pkl")

with open('val_dataset.pkl', 'wb') as f:
    pickle.dump(val_dataset, f)
logging.info("Validation dataset saved to val_dataset.pkl")
