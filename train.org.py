import os
import sentencepiece as spm
import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List
from torch import Tensor
import torch.nn as nn
from torch.nn import Transformer
import math
from torch.utils.data import Dataset, DataLoader
from timeit import default_timer as timer

# Constants
SRC_LANGUAGE = 'en'
TGT_LANGUAGE = 'ja'
BATCH_SIZE = 128
NUM_EPOCHS = 35
GRADIENT_ACCUMULATION_STEPS = 2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
current_dir = os.getcwd()
sp_model_en_path = os.path.join(current_dir, 'sp_en.model')
sp_model_ja_path = os.path.join(current_dir, 'sp_ja.model')
vocab_en_path = os.path.join(current_dir, 'vocab_en.txt')
vocab_ja_path = os.path.join(current_dir, 'vocab_ja.txt')
corpus_en_train_path = os.path.join(current_dir, 'corpus_en_train.txt')
corpus_ja_train_path = os.path.join(current_dir, 'corpus_ja_train.txt')
corpus_en_valid_path = os.path.join(current_dir, 'corpus_en_valid.txt')
corpus_ja_valid_path = os.path.join(current_dir, 'corpus_ja_valid.txt')
model_arch_path = os.path.join(current_dir, 'model_arch.pth')
best_model_path = os.path.join(current_dir, 'best_model.pth')
log_filename = os.path.join(current_dir, 'training_log.txt')

# Load SentencePiece model
sp_en = spm.SentencePieceProcessor()
sp_ja = spm.SentencePieceProcessor()

if not os.path.exists(sp_model_en_path) or not os.path.exists(sp_model_ja_path):
    raise FileNotFoundError("SentencePiece model files not found. Please ensure 'sp_en.model' and 'sp_ja.model' are in the current directory.")

sp_en.load(sp_model_en_path)
sp_ja.load(sp_model_ja_path)

def sp_splitter(sp_model):
    def get_subwords(text):
        return sp_model.encode_as_pieces(text)
    return get_subwords

text_to_subwords = {
    SRC_LANGUAGE: sp_splitter(sp_en),
    TGT_LANGUAGE: sp_splitter(sp_ja)
}

# Function to load vocabulary from a text file and handle special tokens
def load_vocab(path):
    vocab = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                token, idx = parts
                vocab[token] = int(idx)
            else:
                print(f"Skipping invalid line: {line.strip()}")
    # Adding special tokens to vocab
    special_tokens = {'<unk>': 0, '<pad>': 1, '<bos>': 2, '<eos>': 3}
    for token, idx in special_tokens.items():
        vocab[token] = idx
    return vocab

# Load vocabulary
subwords_to_ids = {
    SRC_LANGUAGE: load_vocab(vocab_en_path),
    TGT_LANGUAGE: load_vocab(vocab_ja_path)
}

# Constants for BOS, EOS, and PAD tokens
BOS_IDX = subwords_to_ids[SRC_LANGUAGE]['<bos>']
EOS_IDX = subwords_to_ids[SRC_LANGUAGE]['<eos>']
PAD_IDX = subwords_to_ids[SRC_LANGUAGE]['<pad>']

def add_bos_eos(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]), torch.tensor(token_ids), torch.tensor([EOS_IDX])))

def sequential_transforms(input, language):
    tok_out = text_to_subwords[language](input)
    voc_out = [subwords_to_ids[language].get(tok, PAD_IDX) for tok in tok_out]
    ten_out = add_bos_eos(voc_out)
    return ten_out

text_to_tensor = {
    SRC_LANGUAGE: lambda input: sequential_transforms(input, SRC_LANGUAGE),
    TGT_LANGUAGE: lambda input: sequential_transforms(input, TGT_LANGUAGE)
}

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
    def __init__(self, num_encoder_layers: int, num_decoder_layers: int, emb_size: int, nhead: int, src_vocab_size: int, tgt_vocab_size: int, dim_feedforward: int = 512, dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(self, src: Tensor, trg: Tensor, src_mask: Tensor, tgt_mask: Tensor, src_padding_mask: Tensor, tgt_padding_mask: Tensor, memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None, src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask)

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    return mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)
    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

class CustomDataset(Dataset):
    def __init__(self, src_path, tgt_path):
        with open(src_path, 'r', encoding='utf-8') as f:
            self.src_data = [line.strip() for line in f]
        with open(tgt_path, 'r', encoding='utf-8') as f:
            self.tgt_data = [line.strip() for line in f]

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        return self.src_data[idx], self.tgt_data[idx]

def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_to_tensor[SRC_LANGUAGE](src_sample.rstrip("\n")))
        tgt_batch.append(text_to_tensor[TGT_LANGUAGE](tgt_sample.rstrip("\n")))
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch

def decode_translation(model, src, tgt):
    model.eval()
    src = src.to(DEVICE)
    tgt = tgt.to(DEVICE)

    tgt_input = tgt[:-1, :]
    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

    memory = model.encode(src, src_mask)
    outs = model.decode(tgt_input, memory, tgt_mask)
    outs = model.generator(outs)
    return outs.argmax(dim=-1)

def print_samples(model, train_dataloader, step):
    model.eval()
    src, tgt = next(iter(train_dataloader))
    src, tgt = src.to(DEVICE), tgt.to(DEVICE)
    decoded_preds = decode_translation(model, src, tgt)
    
    for i in range(min(len(src), 3)):  # print up to 3 samples
        src_sentence = " ".join([sp_en.id_to_piece(int(idx)) for idx in src[:, i] if idx in range(sp_en.get_piece_size()) and idx not in [BOS_IDX, EOS_IDX, PAD_IDX]])
        tgt_sentence = " ".join([sp_ja.id_to_piece(int(idx)) for idx in tgt[:, i] if idx in range(sp_ja.get_piece_size()) and idx not in [BOS_IDX, EOS_IDX, PAD_IDX]])
        pred_sentence = " ".join([sp_ja.id_to_piece(int(idx)) for idx in decoded_preds[:, i] if idx in range(sp_ja.get_piece_size()) and idx not in [BOS_IDX, EOS_IDX, PAD_IDX]])
        print(f"Sample {i+1}:")
        print(f"  Source:      {src_sentence}")
        print(f"  Target:      {tgt_sentence}")
        print(f"  Prediction:  {pred_sentence}")

def train_epoch(model, optimizer, train_dataloader):
    model.train()
    losses = 0
    for step, (src, tgt) in enumerate(train_dataloader):
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)
        tgt_input = tgt[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss = loss / GRADIENT_ACCUMULATION_STEPS
        loss.backward()
        losses += loss.item()
        if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()
        if (step + 1) % 50 == 0:
            print(f"Step {step + 1}, Loss: {loss.item()}")
            print_samples(model, train_dataloader, step)
    return losses / len(list(train_dataloader))

def evaluate(model, val_dataloader):
    model.eval()
    losses = 0
    for src, tgt in val_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)
        tgt_input = tgt[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()
    return losses / len(list(val_dataloader))

# Model architecture configuration
model_arch = {
    'NUM_ENCODER_LAYERS': 3,
    'NUM_DECODER_LAYERS': 3,
    'EMB_SIZE': 512,
    'NHEAD': 8,
    'SRC_VOCAB_SIZE': len(subwords_to_ids[SRC_LANGUAGE]),
    'TGT_VOCAB_SIZE': len(subwords_to_ids[TGT_LANGUAGE]),
    'FFN_HID_DIM': 512
}
torch.save(model_arch, model_arch_path)

# Initialize model, loss function, and optimizer
transformer = Seq2SeqTransformer(
    num_encoder_layers=model_arch['NUM_ENCODER_LAYERS'],
    num_decoder_layers=model_arch['NUM_DECODER_LAYERS'],
    emb_size=model_arch['EMB_SIZE'],
    nhead=model_arch['NHEAD'],
    src_vocab_size=model_arch['SRC_VOCAB_SIZE'],
    tgt_vocab_size=model_arch['TGT_VOCAB_SIZE'],
    dim_feedforward=model_arch['FFN_HID_DIM']
)
torch.manual_seed(0)
for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(DEVICE)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

# Load training and validation datasets
train_dataset = CustomDataset(corpus_en_train_path, corpus_ja_train_path)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_dataset = CustomDataset(corpus_en_valid_path, corpus_ja_valid_path)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# Training loop
best_val_loss = float('inf')
for epoch in range(1, NUM_EPOCHS + 1):
    torch.cuda.empty_cache()
    start_time = timer()
    train_loss = train_epoch(transformer, optimizer, train_dataloader)
    end_time = timer()
    val_loss = evaluate(transformer, val_dataloader)
    log_message = f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, Epoch time = {(end_time - start_time):.3f}s"
    print(log_message)
    with open(log_filename, "a") as f:
        f.write(log_message + "\n")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(transformer.state_dict(), best_model_path)
        torch.save(transformer.state_dict(), f"best_model_epoch={epoch}_valloss={val_loss:.3f}.pth")
