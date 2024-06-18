import os
import pickle
import logging
import torch
from datasets import load_dataset
import sentencepiece as spm
from torch.utils.data import Dataset
import random

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

SRC_LANGUAGE = 'en'
TGT_LANGUAGE = 'ja'

# Disable torchtext deprecation warnings
import torchtext
torchtext.disable_torchtext_deprecation_warning()

# Paths
corpus_dir_path = './corpus'
SP_dir_path = './sp_model'
vocab_dir_path = './vocab'

# Create directories if not exist
os.makedirs(corpus_dir_path, exist_ok=True)
os.makedirs(SP_dir_path, exist_ok=True)
os.makedirs(vocab_dir_path, exist_ok=True)

# Corpus file paths
corpus_en_path = os.path.join(corpus_dir_path, 'corpus_en_train.txt')
corpus_ja_path = os.path.join(corpus_dir_path, 'corpus_ja_train.txt')
corpus_enja_path = os.path.join(corpus_dir_path, 'corpus_enja_train.txt')

# SentencePiece model prefix
sp_model_prefix = os.path.join(SP_dir_path, 'sp')

# Vocabulary size
vocab_size = 30000

# Function to save text data to a file
def save_text_data(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for line in data:
            f.write(line + '\n')

# Function to train SentencePiece model
def train_sentencepiece_model(corpus_path, model_prefix, vocab_size=30000):
    sp_command = f'--input={corpus_path} --model_prefix={model_prefix} --vocab_size={vocab_size} --character_coverage=0.995 --model_type=bpe'
    spm.SentencePieceTrainer.train(sp_command)

# Load and select dataset
def load_and_select_dataset(src_lang, tgt_lang):
    logging.info("Loading dataset")
    dataset = load_dataset('opus100', f'{src_lang}-{tgt_lang}')
    logging.info("Selecting samples from dataset")
    selected_samples = dataset['train']
    return selected_samples, dataset['validation']

# Save English and Japanese sentences to separate files
logging.info("Loading and selecting dataset")
train_data, val_data = load_and_select_dataset(SRC_LANGUAGE, TGT_LANGUAGE)
logging.info(f"Loaded {len(train_data)} training samples and {len(val_data)} validation samples")

save_text_data([sample['translation'][SRC_LANGUAGE] for sample in train_data], corpus_en_path)
save_text_data([sample['translation'][TGT_LANGUAGE] for sample in train_data], corpus_ja_path)
save_text_data([sample['translation'][SRC_LANGUAGE] + '\t' + sample['translation'][TGT_LANGUAGE] for sample in train_data], corpus_enja_path)

# Train SentencePiece models if not already trained
if not os.path.exists(sp_model_prefix + '.model'):
    logging.info("Training SentencePiece model")
    train_sentencepiece_model(corpus_enja_path, sp_model_prefix, vocab_size)

# Load SentencePiece models
sp = spm.SentencePieceProcessor()
sp.load(sp_model_prefix + '.model')

# Tokenization functions
def sp_splitter(sp_model):
    def get_subwords(text):
        return sp_model.encode_as_pieces(text)
    return get_subwords

text_to_subwords = {
    SRC_LANGUAGE: sp_splitter(sp),
    TGT_LANGUAGE: sp_splitter(sp)
}

# Function to build vocabulary from iterator
def build_vocab_from_iterator(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            yield sp.encode_as_pieces(line)

# Special tokens
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

# Build vocabularies
subwords_to_ids = {
    SRC_LANGUAGE: torch.load(os.path.join(vocab_dir_path, 'vocab_en.pth')) if os.path.exists(os.path.join(vocab_dir_path, 'vocab_en.pth')) else build_vocab_from_iterator(corpus_en_path),
    TGT_LANGUAGE: torch.load(os.path.join(vocab_dir_path, 'vocab_ja.pth')) if os.path.exists(os.path.join(vocab_dir_path, 'vocab_ja.pth')) else build_vocab_from_iterator(corpus_ja_path)
}

# Save vocabularies
torch.save(subwords_to_ids[SRC_LANGUAGE], os.path.join(vocab_dir_path, 'vocab_en.pth'))
torch.save(subwords_to_ids[TGT_LANGUAGE], os.path.join(vocab_dir_path, 'vocab_ja.pth'))

# Default to unknown token index
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    subwords_to_ids[ln].set_default_index(UNK_IDX)

# Text transformations
def numericalize(tokens, vocab):
    return [vocab[token] for token in tokens]

def tensor_transform(token_ids: list[int]):
    return torch.tensor([BOS_IDX] + token_ids + [EOS_IDX])

text_transform = {
    SRC_LANGUAGE: lambda text: tensor_transform(numericalize(text_to_subwords[SRC_LANGUAGE](text), subwords_to_ids[SRC_LANGUAGE])),
    TGT_LANGUAGE: lambda text: tensor_transform(numericalize(text_to_subwords[TGT_LANGUAGE](text), subwords_to_ids[TGT_LANGUAGE]))
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

# Create datasets
logging.info("Creating training and validation datasets")
train_dataset = TranslationDataset(train_data)
val_dataset = TranslationDataset(val_data)

# Sample 1000 random examples
sample_indices = random.sample(range(len(train_dataset)), 1000)
random_samples = [train_dataset[i] for i in sample_indices]

# Print the 1000 random examples of raw, encoded, and decoded sentences
for i, (src_text, tgt_text, src_sample, tgt_sample) in enumerate(random_samples):
    decoded_src = " ".join([sp.id_to_piece(idx) for idx in src_sample if idx not in [BOS_IDX, EOS_IDX]])
    decoded_tgt = " ".join([sp.id_to_piece(idx) for idx in tgt_sample if idx not in [BOS_IDX, EOS_IDX]])
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
