import os
import logging
import pickle
from typing import Iterable, List
from datasets import load_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, Vocab
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

SRC_LANGUAGE = 'en'
TGT_LANGUAGE = 'ja'
token_transform = {}
vocab_transform = {}

token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')
token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language='ja_core_news_sm')

def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
    language_index = {SRC_LANGUAGE: 'translation', TGT_LANGUAGE: 'translation'}
    for data_sample in data_iter:
        yield token_transform[language](data_sample[language_index[language]][language])

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

cache_dir = 'cache'
os.makedirs(cache_dir, exist_ok=True)

def save_vocab(vocab: Vocab, path: str):
    with open(path, 'wb') as f:
        pickle.dump(vocab, f)

def load_vocab(path: str) -> Vocab:
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_and_select_dataset(src_lang, tgt_lang, num_samples=1000):
    dataset = load_dataset('opus100', f'{src_lang}-{tgt_lang}')
    selected_samples = dataset['train'].select(range(num_samples))
    return selected_samples

# Load and select 1000 samples from the dataset
dataset = load_and_select_dataset(SRC_LANGUAGE, TGT_LANGUAGE)

for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    vocab_path = os.path.join(cache_dir, f'{ln}_vocab.pkl')
    if os.path.exists(vocab_path):
        logging.info(f"Loading cached vocabulary for {ln}...")
        vocab_transform[ln] = load_vocab(vocab_path)
    else:
        logging.info(f"Building vocabulary for {ln}...")
        start_time = timer()
        vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(dataset, ln),
                                                        min_freq=1,
                                                        specials=special_symbols,
                                                        special_first=True)
        end_time = timer()
        logging.info(f"Vocabulary for {ln} built in {end_time - start_time:.3f}s")
        save_vocab(vocab_transform[ln], vocab_path)

for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    vocab_transform[ln].set_default_index(UNK_IDX)

class TranslationDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_sample = text_transform[SRC_LANGUAGE](self.data[idx]['translation'][SRC_LANGUAGE].rstrip("\n")).tolist()
        tgt_sample = text_transform[TGT_LANGUAGE](self.data[idx]['translation'][TGT_LANGUAGE].rstrip("\n")).tolist()
        return src_sample, tgt_sample

def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(torch.tensor(src_sample))
        tgt_batch.append(torch.tensor(tgt_sample))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch

torch.manual_seed(0)

def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))

text_transform = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    text_transform[ln] = sequential_transforms(token_transform[ln], vocab_transform[ln], tensor_transform)

train_dataset = TranslationDataset(dataset)
train_dataloader = DataLoader(train_dataset, batch_size=1, collate_fn=collate_fn)

if __name__ == "__main__":
    # Here you can save the DataLoader or perform any additional processing if needed
    pass
