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
from typing import List
from tqdm import tqdm
import sentencepiece as spm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

SRC_LANGUAGE = 'en'
TGT_LANGUAGE = 'ja'
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
BATCH_SIZE = 32
GRADIENT_ACCUMULATION_STEPS = 2
NUM_EPOCHS = 36
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 2048
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6
DROPOUT = 0.3
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load SentencePiece models
src_sp = spm.SentencePieceProcessor(model_file='spm_en.model')
tgt_sp = spm.SentencePieceProcessor(model_file='spm_ja.model')

# Transformer model definition
class Seq2SeqTransformer(nn.Module):
    def __init__(self, num_encoder_layers: int, num_decoder_layers: int, emb_size: int, nhead: int,
                 src_vocab_size: int, tgt_vocab_size: int, dim_feedforward: int = 512, dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = nn.Embedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout)

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Tensor, tgt_mask: Tensor,
                src_padding_mask: Tensor, tgt_padding_mask: Tensor, memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        memory = self.transformer.encoder(src_emb, src_mask, src_padding_mask)
        outs = self.transformer.decoder(tgt_emb, memory, tgt_mask, None, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor, src_padding_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(self.src_tok_emb(src)), src_mask, src_padding_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor, memory_key_padding_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask, None, memory_key_padding_mask)

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
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_text, tgt_text, src_sample, tgt_sample in batch:
        src_batch.append(torch.tensor(src_sample))
        tgt_batch.append(torch.tensor(tgt_sample))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch

# Define the TranslationDataset class
class TranslationDataset(Dataset):
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
        
        src_sample = [BOS_IDX] + src_tokens + [EOS_IDX]
        tgt_sample = [BOS_IDX] + tgt_tokens + [EOS_IDX]
        
        return src_text, tgt_text, src_sample, tgt_sample

# Model training and evaluation
def train_epoch(model, optimizer, train_dataloader, scaler, scheduler):
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

        with torch.cuda.amp.autocast():
            logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
            tgt_out = tgt[1:, :]
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

        scaler.scale(loss).backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        if (i + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        losses += loss.item()

        if (i + 1) % 500 == 0:  # Log every 500 iterations
            torch.cuda.empty_cache()  # Clear GPU cache
            # Decode a single sample for logging
            src_sentence = src_sp.decode([idx for idx in src[:, 0].tolist() if idx != BOS_IDX and idx != EOS_IDX and idx != PAD_IDX])
            tgt_sentence = tgt_sp.decode([idx for idx in tgt[:, 0].tolist() if idx != BOS_IDX and idx != EOS_IDX and idx != PAD_IDX])
            pred_tokens = logits.argmax(dim=-1)[:, 0].tolist()
            pred_sentence = tgt_sp.decode([idx for idx in pred_tokens if idx != BOS_IDX and idx != EOS_IDX and idx != PAD_IDX])
            
            # Log the decoded sentences and loss
            logging.info(f"Iteration {i+1}/{len(train_dataloader)}: Loss = {loss.item()}")
            logging.info(f"Src: {src_sentence}")
            logging.info(f"True Tgt: {tgt_sentence}")
            logging.info(f"Pred Tgt: {pred_sentence}")
        
        # Update progress bar description with the latest loss
        progress_bar.set_description(f"Training (loss {loss.item():.4f})")

    scheduler.step()
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
            src_sentence = src_sp.decode([idx for idx in src[:, 0].tolist() if idx != BOS_IDX and idx != EOS_IDX and idx != PAD_IDX])
            tgt_sentence = tgt_sp.decode([idx for idx in tgt[:, 0].tolist() if idx != BOS_IDX and idx != EOS_IDX and idx != PAD_IDX])
            pred_tokens = logits.argmax(dim=-1)[:, 0].tolist()
            pred_sentence = tgt_sp.decode([idx for idx in pred_tokens if idx != BOS_IDX and idx != EOS_IDX and idx != PAD_IDX])
            
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
                                     src_sp.get_piece_size(), tgt_sp.get_piece_size(), FFN_HID_DIM, DROPOUT)
    transformer = transformer.to(DEVICE)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.AdamW(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    scaler = torch.cuda.amp.GradScaler()  # Mixed precision training
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-5)  # Cosine annealing scheduler

    # Training loop
    model_dir = "saved_models"
    os.makedirs(model_dir, exist_ok=True)

    for epoch in range(1, NUM_EPOCHS + 1):
        logging.info(f"Starting epoch {epoch}/{NUM_EPOCHS}")
        start_time = timer()
        train_loss = train_epoch(transformer, optimizer, train_dataloader, scaler, scheduler)
        val_loss = evaluate(transformer, val_dataloader)
        end_time = timer()
        epoch_time = end_time - start_time

        # Log and save model
        logging.info(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, Epoch time = {epoch_time:.3f}s")
        model_path = os.path.join(model_dir, f"transformer_epoch_{epoch}.pth")
        torch.save(transformer.state_dict(), model_path)

        logging.info(f"Finished epoch {epoch}/{NUM_EPOCHS}")
