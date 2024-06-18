import logging
import os
import pickle
import torch
import math
import time
from torch.utils.data import DataLoader, Dataset
import spacy
from torch.nn import Embedding, TransformerDecoderLayer, TransformerDecoder, TransformerEncoderLayer, TransformerEncoder
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torchtext.vocab import Vocab
from torch.nn.init import xavier_uniform_
from typing import Optional

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load SpaCy models
JA = spacy.load("ja_core_news_md")
EN = spacy.load("en_core_web_md")

# Tokenization functions
def tokenize_ja(sentence):
    return [tok.text for tok in JA.tokenizer(sentence)]

def tokenize_en(sentence):
    return [tok.text for tok in EN.tokenizer(sentence)]

# Special tokens
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

# Load vocabularies from pickle files
def load_vocab(path: str):
    with open(path, 'rb') as f:
        vocab = pickle.load(f)
    return vocab

en_vocab = load_vocab('en_vocab.pkl')
ja_vocab = load_vocab('ja_vocab.pkl')

def numericalize_en(tokens: list[str]):
    return [en_vocab[token] for token in tokens]

def numericalize_ja(tokens: list[str]):
    return [ja_vocab[token] for token in tokens]

def tensor_transform(token_ids: list[int]):
    return torch.tensor([BOS_IDX] + token_ids + [EOS_IDX])

def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# Text transformations
text_transform = {
    'en': sequential_transforms(tokenize_en, numericalize_en, tensor_transform),
    'ja': sequential_transforms(tokenize_ja, numericalize_ja, tensor_transform)
}

# Load datasets from pickle files
def load_dataset(path: str):
    with open(path, 'rb') as f:
        dataset = pickle.load(f)
    return dataset

class TranslationDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_text = self.data[idx]['translation']['ja'].rstrip("\n")
        tgt_text = self.data[idx]['translation']['en'].rstrip("\n")
        src_sample = text_transform['ja'](src_text)
        tgt_sample = text_transform['en'](tgt_text)
        return src_text, tgt_text, src_sample, tgt_sample

train_dataset = load_dataset('train_dataset.pkl')
val_dataset = load_dataset('val_dataset.pkl')

# Create datasets
train_dataset = TranslationDataset(train_dataset.data)
val_dataset = TranslationDataset(val_dataset.data)

# Collate function for padding
def collate_fn(batch):
    src_batch, tgt_batch = zip(*[(src, tgt) for _, _, src, tgt in batch])
    src_batch = torch.nn.utils.rnn.pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = torch.nn.utils.rnn.pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch

# Create data loaders
batch_size = 12
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout: float = 0.1, max_len: int = 5000) -> None:
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x) -> torch.Tensor:
        x = x * math.sqrt(self.d_model)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Transformer(nn.Module):

    def __init__(self, 
                 src_vocab_size: int = 3000,
                 d_model: int = 512,
                 nhead: int = 8,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.5,
                 activation: str = 'relu',
                 num_layers: int = 6,
                 tgt_vocab_size: int = 3000
                 ) -> None:
        super(Transformer, self).__init__()
        self.src_embedd = nn.Embedding(src_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers, norm=encoder_norm)

        self.tgt_embedd = nn.Embedding(tgt_vocab_size, d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout=dropout, activation=activation)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers, norm=decoder_norm)

        self.out = nn.Linear(d_model, tgt_vocab_size)
        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, 
                src: torch.Tensor,
                tgt: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:

        src = self.src_embedd(src)
        src = self.pos_encoder(src)
        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        tgt = self.tgt_embedd(tgt)
        tgt = self.pos_encoder(tgt)
        out = self.decoder(tgt,
                            memory,
                            tgt_mask=tgt_mask,
                            memory_mask=memory_mask,
                            tgt_key_padding_mask=tgt_key_padding_mask,
                            memory_key_padding_mask=memory_key_padding_mask
                            )
        out = self.out(out)

        return out

def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len).to(device)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


src_vocab_size = len(ja_vocab)
tgt_vocab_size = len(en_vocab)

model = Transformer(src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size)
model.to(device)

criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=0.0001)

def train(epochs: int = 300) -> None:
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        start_time = time.time()
        for step, data in enumerate(train_dataloader):
            src, tgt = data
            src = src.to(device)
            tgt = tgt.to(device)

            tgt_input = tgt[:-1, :]
            targets = tgt[1:, :].contiguous().view(-1)

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

            # Forward pass
            pred = model(src, tgt_input, src_mask, tgt_mask, None, src_padding_mask, tgt_padding_mask, src_padding_mask)
            pred = pred.contiguous().view(-1, pred.size(-1))

            # Calculate loss
            loss = criterion(pred, targets)

            # Backpropagation
            optim.zero_grad()
            loss.backward()
            optim.step()

            # Update loss
            total_loss += loss.item() / batch_size

            # Log every 50 steps
            if (step + 1) % 50 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{step+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}')

                # Generate example prediction
                example_src, example_tgt = src[:, 0], tgt[:, 0]
                example_src_text = " ".join([ja_vocab.get_itos()[i.item()] for i in example_src if i.item() not in [PAD_IDX, BOS_IDX, EOS_IDX]])
                example_tgt_text = " ".join([en_vocab.get_itos()[i.item()] for i in example_tgt if i.item() not in [PAD_IDX, BOS_IDX, EOS_IDX]])
                
                # Simple greedy decoding
                with torch.no_grad():
                    memory = model.encoder(model.src_embedd(example_src.unsqueeze(1)))
                    ys = torch.ones(1, 1).fill_(BOS_IDX).type(torch.long).to(device)
                    for i in range(100):  # limiting the max length of the prediction
                        tgt_mask = generate_square_subsequent_mask(ys.size(0)).type(torch.bool).to(device)
                        out = model.decoder(model.tgt_embedd(ys), memory, tgt_mask)
                        prob = model.out(out[-1, :])
                        _, next_word = torch.max(prob, dim=1)
                        next_word = next_word.item()
                        ys = torch.cat([ys, torch.ones(1, 1).type_as(example_src.data).fill_(next_word)], dim=0)
                        if next_word == EOS_IDX:
                            break

                    example_prediction = " ".join([en_vocab.get_itos()[i.item()] for i in ys.squeeze() if i.item() not in [PAD_IDX, BOS_IDX, EOS_IDX]])
                    
                print(f"Example Source: {example_src_text}")
                print(f"Example Target: {example_tgt_text}")
                print(f"Example Prediction: {example_prediction}")
                print()

        end_time = time.time()
        print(f'Epoch [{epoch+1}/{epochs}], Total Loss: {total_loss:.4f}, Elapsed Time: {end_time - start_time:.2f}s')

if __name__ == "__main__":
    train(epochs=10)
