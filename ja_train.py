import torchtext
torchtext.disable_torchtext_deprecation_warning()

import os
import math
import logging
import torch
import torch.nn as nn
from torch.nn import Transformer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch import Tensor
from timeit import default_timer as timer
import pickle
from torchtext.vocab import Vocab
from torchtext.data.utils import get_tokenizer
from typing import List
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

SRC_LANGUAGE = 'en'
TGT_LANGUAGE = 'ja'
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 2
NUM_EPOCHS = 36
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load vocabularies
cache_dir = 'cache'
def load_vocab(path: str) -> Vocab:
    with open(path, 'rb') as f:
        return pickle.load(f)

vocab_transform = {
    SRC_LANGUAGE: load_vocab(os.path.join(cache_dir, f'{SRC_LANGUAGE}_vocab.pkl')),
    TGT_LANGUAGE: load_vocab(os.path.join(cache_dir, f'{TGT_LANGUAGE}_vocab.pkl'))
}

for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    vocab_transform[ln].set_default_index(UNK_IDX)

# Tokenizers
token_transform = {
    SRC_LANGUAGE: get_tokenizer('spacy', language='en_core_web_sm'),
    TGT_LANGUAGE: get_tokenizer('spacy', language='ja_core_news_sm')
}

# Transformer model definition
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

class Seq2SeqTransformer(nn.Module):
    def __init__(self, num_encoder_layers: int, num_decoder_layers: int, emb_size: int, nhead: int,
                 src_vocab_size: int, tgt_vocab_size: int, dim_feedforward: int = 512, dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(self, src: Tensor, trg: Tensor, src_mask: Tensor, tgt_mask: Tensor,
                src_padding_mask: Tensor, tgt_padding_mask: Tensor, memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask)

# Masking functions
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

# Data preparation
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

# Model training and evaluation
def train_epoch(model, optimizer, train_dataloader):
    model.train()
    losses = 0
    start_time = timer()
    optimizer.zero_grad()
    progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="Training")
    for i, (src, tgt) in progress_bar:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        if (i + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()

        losses += loss.item()

        if (i + 1) % 500 == 0:  # Log every 500 iterations
            # Decode a single sample for logging
            src_sentence = " ".join([vocab_transform[SRC_LANGUAGE].lookup_token(idx) for idx in src[:, 0].tolist() if idx != PAD_IDX])
            tgt_sentence = " ".join([vocab_transform[TGT_LANGUAGE].lookup_token(idx) for idx in tgt[:, 0].tolist() if idx != PAD_IDX])
            pred_tokens = logits.argmax(dim=-1)[:, 0].tolist()
            pred_sentence = " ".join([vocab_transform[TGT_LANGUAGE].lookup_token(idx) for idx in pred_tokens if idx != PAD_IDX])
            
            # Log the decoded sentences and loss
            logging.info(f"Iteration {i+1}/{len(train_dataloader)}: Loss = {loss.item()}")
            logging.info(f"Src: {src_sentence}")
            logging.info(f"True Tgt: {tgt_sentence}")
            logging.info(f"Pred Tgt: {pred_sentence}")
        
        # Update progress bar description with the latest loss
        progress_bar.set_description(f"Training (loss {loss.item():.4f})")

    end_time = timer()
    logging.info(f"Training epoch time: {end_time - start_time:.3f}s")

    return losses / len(train_dataloader)

def evaluate(model, val_dataloader):
    model.eval()
    losses = 0
    start_time = timer()
    with torch.no_grad():
        for src, tgt in tqdm(val_dataloader, desc="Validation"):
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)

            tgt_input = tgt[:-1, :]

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

            logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

            tgt_out = tgt[1:, :]
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            losses += loss.item()

            # Decode a single sample for logging
            src_sentence = " ".join([vocab_transform[SRC_LANGUAGE].lookup_token(idx) for idx in src[:, 0].tolist() if idx != PAD_IDX])
            tgt_sentence = " ".join([vocab_transform[TGT_LANGUAGE].lookup_token(idx) for idx in tgt[:, 0].tolist() if idx != PAD_IDX])
            pred_tokens = logits.argmax(dim=-1)[:, 0].tolist()
            pred_sentence = " ".join([vocab_transform[TGT_LANGUAGE].lookup_token(idx) for idx in pred_tokens if idx != PAD_IDX])
            
            logging.info(f"Validation - Src: {src_sentence}")
            logging.info(f"Validation - True Tgt: {tgt_sentence}")
            logging.info(f"Validation - Pred Tgt: {pred_sentence}")

    end_time = timer()
    logging.info(f"Validation epoch time: {end_time - start_time:.3f}s")

    return losses / len(val_dataloader)

if __name__ == "__main__":
    # Load datasets
    logging.info("Loading datasets")
    with open('train_dataset.pkl', 'rb') as f:
        train_dataset = pickle.load(f)
    with open('val_dataset.pkl', 'rb') as f:
        val_dataset = pickle.load(f)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    # Initialize model, loss function, and optimizer
    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, NHEAD, 
                                     len(vocab_transform[SRC_LANGUAGE]), len(vocab_transform[TGT_LANGUAGE]), FFN_HID_DIM)
    transformer = transformer.to(DEVICE)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    # Training loop
    model_dir = "saved_models"
    os.makedirs(model_dir, exist_ok=True)

    for epoch in range(1, NUM_EPOCHS + 1):
        logging.info(f"Starting epoch {epoch}/{NUM_EPOCHS}")
        start_time = timer()
        train_loss = train_epoch(transformer, optimizer, train_dataloader)
        val_loss = evaluate(transformer, val_dataloader)
        end_time = timer()
        epoch_time = end_time - start_time

        # Log and save model
        logging.info(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, Epoch time = {epoch_time:.3f}s")
        model_path = os.path.join(model_dir, f"transformer_epoch_{epoch}.pth")
        torch.save(transformer.state_dict(), model_path)

        logging.info(f"Finished epoch {epoch}/{NUM_EPOCHS}")
