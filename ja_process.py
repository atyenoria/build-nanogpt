import os
import pickle
import logging
import torch
from datasets import load_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, Vocab

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

SRC_LANGUAGE = 'en'
TGT_LANGUAGE = 'ja'
token_transform = {}
vocab_transform = {}

token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')
token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language='ja_core_news_sm')

def yield_tokens(data_iter, language):
    language_index = {SRC_LANGUAGE: 'translation', TGT_LANGUAGE: 'translation'}
    for data_sample in data_iter:
        tokens = token_transform[language](data_sample[language_index[language]][language])
        logging.debug(f"Tokenizing {data_sample[language_index[language]][language]} -> {tokens}")
        yield tokens

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

def load_and_select_dataset(src_lang, tgt_lang, num_samples=50000):
    logging.info("Loading dataset")
    dataset = load_dataset('opus100', f'{src_lang}-{tgt_lang}')
    logging.info("Selecting samples from dataset")
    selected_samples = dataset['train'].select(range(num_samples))
    return selected_samples, dataset['validation']

# Load and select 50,000 samples from the dataset
logging.info("Loading and selecting dataset")
train_data, val_data = load_and_select_dataset(SRC_LANGUAGE, TGT_LANGUAGE, num_samples=50000)
logging.info(f"Loaded {len(train_data)} training samples and {len(val_data)} validation samples")

for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    vocab_path = os.path.join(cache_dir, f'{ln}_vocab.pkl')
    if os.path.exists(vocab_path):
        vocab_transform[ln] = load_vocab(vocab_path)
    else:
        logging.info(f"Building vocabulary for {ln}")
        vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_data, ln),
                                                        min_freq=1,
                                                        specials=special_symbols,
                                                        special_first=True)
        save_vocab(vocab_transform[ln], vocab_path)

for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    vocab_transform[ln].set_default_index(UNK_IDX)

    # Log vocabulary details
    logging.info(f"{ln} vocabulary size: {len(vocab_transform[ln])}")
    logging.info(f"Sample {ln} vocabulary: {vocab_transform[ln].get_itos()[:10]}")

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

text_transform = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    text_transform[ln] = sequential_transforms(token_transform[ln], vocab_transform[ln], tensor_transform)

class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_text = self.data[idx]['translation'][SRC_LANGUAGE].rstrip("\n")
        tgt_text = self.data[idx]['translation'][TGT_LANGUAGE].rstrip("\n")
        
        logging.debug(f"Original Src: {src_text}")
        logging.debug(f"Original Tgt: {tgt_text}")
        
        src_sample = text_transform[SRC_LANGUAGE](src_text).tolist()
        tgt_sample = text_transform[TGT_LANGUAGE](tgt_text).tolist()
        
        logging.debug(f"Transformed Src: {src_sample}")
        logging.debug(f"Transformed Tgt: {tgt_sample}")
        
        return src_sample, tgt_sample

# Create datasets
logging.info("Creating training and validation datasets")
train_dataset = TranslationDataset(train_data)
val_dataset = TranslationDataset(val_data)

# Log some sample data
for i in range(3):
    src_sample, tgt_sample = train_dataset[i]
    logging.info(f"Sample {i} - Src: {src_sample}, Tgt: {tgt_sample}")

# Save datasets to pickle files
with open('train_dataset.pkl', 'wb') as f:
    pickle.dump(train_dataset, f)
logging.info("Training dataset saved to train_dataset.pkl")

with open('val_dataset.pkl', 'wb') as f:
    pickle.dump(val_dataset, f)
logging.info("Validation dataset saved to val_dataset.pkl")
