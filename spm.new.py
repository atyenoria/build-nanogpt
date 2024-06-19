import os
import pickle
import logging
import torch
from datasets import load_dataset
from torchtext.vocab import build_vocab_from_iterator, Vocab
import random
import sentencepiece as spm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

SRC_LANGUAGE = 'en'
TGT_LANGUAGE = 'ja'

# Disable torchtext deprecation warnings
import torchtext
torchtext.disable_torchtext_deprecation_warning()

# Train SentencePiece models if not already trained
def train_sentencepiece_model(data, model_prefix, vocab_size=32000):
    input_file = f"{model_prefix}.txt"
    with open(input_file, "w", encoding="utf-8") as f:
        for sample in data:
            f.write(sample + "\n")
    
    spm.SentencePieceTrainer.train(input=input_file, model_prefix=model_prefix, vocab_size=vocab_size, model_type='bpe')

if not os.path.exists("spm_en.model"):
    dataset = load_dataset('opus100', f'{SRC_LANGUAGE}-{TGT_LANGUAGE}')
    train_data = dataset['train'].select(range(50000))
    train_sentencepiece_model([sample['translation'][SRC_LANGUAGE] for sample in train_data], "spm_en")
    train_sentencepiece_model([sample['translation'][TGT_LANGUAGE] for sample in train_data], "spm_ja")

# Load SentencePiece models
src_sp = spm.SentencePieceProcessor(model_file='spm_en.model')
tgt_sp = spm.SentencePieceProcessor(model_file='spm_ja.model')

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

cache_dir = 'cache'
os.makedirs(cache_dir, exist_ok=True)

def save_vocab(vocab: Vocab, path: str):
    with open(path, 'wb') as f:
        pickle.dump(vocab, f)
    logging.info(f"Vocabulary saved to {path}")

def load_vocab(path: str) -> Vocab:
    with open(path, 'rb') as f:
        vocab = pickle.load(f)
    logging.info(f"Vocabulary loaded from {path}")
    return vocab

def load_and_select_dataset(src_lang, tgt_lang, num_samples):
    logging.info("Loading dataset")
    dataset = load_dataset('opus100', f'{src_lang}-{tgt_lang}')
    logging.info("Selecting samples from dataset")
    selected_samples = dataset['train'].select(range(num_samples))
    return selected_samples, dataset['validation']

# Load and select 50,000 samples from the dataset
logging.info("Loading and selecting dataset")
train_data, val_data = load_and_select_dataset(SRC_LANGUAGE, TGT_LANGUAGE, num_samples=100000)
logging.info(f"Loaded {len(train_data)} training samples and {len(val_data)} validation samples")

def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

def tensor_transform(token_ids: list[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))

text_transform = {
    SRC_LANGUAGE: sequential_transforms(lambda txt: src_sp.encode(txt, out_type=int), tensor_transform),
    TGT_LANGUAGE: sequential_transforms(lambda txt: tgt_sp.encode(txt, out_type=int), tensor_transform)
}

class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_text = self.data[idx]['translation'][SRC_LANGUAGE].rstrip("\n")
        tgt_text = self.data[idx]['translation'][TGT_LANGUAGE].rstrip("\n")
        
        # Tokenize the raw text
        src_tokens = src_sp.encode(src_text, out_type=int)
        tgt_tokens = tgt_sp.encode(tgt_text, out_type=int)
        
        logging.info(f"Original Src: {src_text}")
        logging.info(f"Tokenized Src: {src_tokens}")
        logging.info(f"Original Tgt: {tgt_text}")
        logging.info(f"Tokenized Tgt: {tgt_tokens}")
        
        src_sample = text_transform[SRC_LANGUAGE](src_text).tolist()
        tgt_sample = text_transform[TGT_LANGUAGE](tgt_text).tolist()
        
        return src_text, tgt_text, src_sample, tgt_sample

# Create datasets
logging.info("Creating training and validation datasets")
train_dataset = TranslationDataset(train_data)
val_dataset = TranslationDataset(val_data)

# Sample 1000 random examples
sample_indices = random.sample(range(len(train_dataset)), 1000)
random_samples = [train_dataset[i] for i in sample_indices]

# Print the 1000 random examples of raw, encoded, and decoded sentences
for i, (src_text, tgt_text, src_sample, tgt_sample) in enumerate(random_samples):
    decoded_src = src_sp.decode([idx for idx in src_sample if idx != BOS_IDX and idx != EOS_IDX])
    decoded_tgt = tgt_sp.decode([idx for idx in tgt_sample if idx != BOS_IDX and idx != EOS_IDX])
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
