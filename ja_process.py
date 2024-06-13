import os
import pickle
import torch
from datasets import load_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, Vocab

SRC_LANGUAGE = 'en'
TGT_LANGUAGE = 'ja'
token_transform = {}
vocab_transform = {}

token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')
token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language='ja_core_news_sm')

def yield_tokens(data_iter, language):
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
    return selected_samples, dataset['validation']

# Load and select 1000 samples from the dataset
train_data, val_data = load_and_select_dataset(SRC_LANGUAGE, TGT_LANGUAGE, num_samples=1000)

for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    vocab_path = os.path.join(cache_dir, f'{ln}_vocab.pkl')
    if os.path.exists(vocab_path):
        vocab_transform[ln] = load_vocab(vocab_path)
    else:
        vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_data, ln),
                                                        min_freq=1,
                                                        specials=special_symbols,
                                                        special_first=True)
        save_vocab(vocab_transform[ln], vocab_path)

for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    vocab_transform[ln].set_default_index(UNK_IDX)

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
        src_sample = text_transform[SRC_LANGUAGE](self.data[idx]['translation'][SRC_LANGUAGE].rstrip("\n")).tolist()
        tgt_sample = text_transform[TGT_LANGUAGE](self.data[idx]['translation'][TGT_LANGUAGE].rstrip("\n")).tolist()
        return src_sample, tgt_sample

train_dataset = TranslationDataset(train_data)
val_dataset = TranslationDataset(val_data)

# Save datasets to pickle files
with open('train_dataset.pkl', 'wb') as f:
    pickle.dump(train_dataset, f)
with open('val_dataset.pkl', 'wb') as f:
    pickle.dump(val_dataset, f)
