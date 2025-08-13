import os
import builtins
import sys 

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import random
import sys
import time
import numpy as np
from tqdm import tqdm

from pytrs import (
    Op,
    VARIABLE_RANGE,
    CONST_OFFSET,
    PAREN_CLOSE,
    PAREN_OPEN,
    node_to_id,
    parse_sexpr,
    tokenize,
    MAX_INT_TOKENS
)

torch.set_float32_matmul_precision("high")

# ------------------------------------------------------------------
# Helper: force-flushing print & memory reporter
# ------------------------------------------------------------------
def print(*args, **kwargs):  
    kwargs["flush"] = True
    builtins.print(*args, **kwargs)



# ------------------------------------------------------------------
# Subclass DDP to allow easier attribute access
# ------------------------------------------------------------------
class AEDDP(DDP):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


# ------------------------------------------------------------------
# DDP setup and device assignment
# ------------------------------------------------------------------
ddp = 0 # int(os.environ.get("RANK", -1)) != -1
deviceids = [0, 1, 2]

if ddp:
    assert torch.cuda.is_available(), "DDP requires CUDA"
    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{deviceids[ddp_rank]}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    master_process = True
    ddp_rank = 0
    ddp_world_size = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def debug_print_memory(tag=""):  
    if master_process:
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated(device) / (1024**2)
            reserved = torch.cuda.memory_reserved(device) / (1024**2)
            print(f"[{tag}] CUDA  allocated={alloc:.2f} MiB  reserved={reserved:.2f} MiB")



if master_process:
    print("Using DDP with world size:", ddp_world_size)
    print("Assigned device:", device)
    
    
debug_print_memory("After device setup")  


# ------------------------------------------------------------------
# Configuration and token setup
# ------------------------------------------------------------------
class Config:
    max_gen_length = 25122

    vocab_size = CONST_OFFSET + MAX_INT_TOKENS + 2 + 1 + 1
    start_token = CONST_OFFSET + MAX_INT_TOKENS
    end_token = CONST_OFFSET + MAX_INT_TOKENS + 1
    pad_token = CONST_OFFSET + MAX_INT_TOKENS + 2

    cls_token = vocab_size
    vocab_size += 1  # include CLS

    d_model = 256
    num_heads = 8
    num_encoder_layers = 4
    num_decoder_layers = 4
    dim_feedforward = 512
    transformer_dropout = 0.2
    max_seq_length = 25200

    batch_size = 16 
    learning_rate = 3e-4
    epochs = 50
    dropout_rate = 0.3

    total_samples = 5000000  

config = Config()

# ------------------------------------------------------------------
# Expression processing helpers
# ------------------------------------------------------------------
def dfs_traverse(expr, depth=0, node_list=None):
    if node_list is None:
        node_list = []
    if isinstance(expr, Op):
        node_list.append((PAREN_OPEN, depth))
        node_list.append((expr, depth))
        for child in expr.args:
            dfs_traverse(child, depth + 1, node_list)
        node_list.append((PAREN_CLOSE, depth))
    else:
        node_list.append((expr, depth))
    return node_list


def flatten_expr(expr):
    node_list = dfs_traverse(expr, 0)
    results = []
    varmap = {}
    intmap = {}
    next_var_id = VARIABLE_RANGE[0]
    next_int_id = CONST_OFFSET
    for node_or_paren, depth in node_list:
        if node_or_paren in (PAREN_OPEN, PAREN_CLOSE):
            nid = node_or_paren
        else:
            nid, next_var_id, next_int_id, _ = node_to_id(
                node_or_paren, varmap, intmap, next_var_id, next_int_id
            )
        results.append({"node_id": nid})
    return results


# ------------------------------------------------------------------
# Positional encoding
# ------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x, positions=None):
        batch_size, seq_len, _ = x.shape
        if positions is None:
            positions = (
                torch.arange(0, seq_len, device=x.device)
                .unsqueeze(0)
                .repeat(batch_size, 1)
            )
        pos_emb = self.pos_embedding(positions)
        return x + pos_emb


# ------------------------------------------------------------------
# Transformer autoencoder
# ------------------------------------------------------------------
class TransformerAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_encoder = PositionalEncoding(
            config.d_model, max_len=config.max_seq_length
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.transformer_dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, config.num_encoder_layers)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.transformer_dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, config.num_decoder_layers)

        self.output_fc = nn.Linear(config.d_model, config.vocab_size)

    @staticmethod
    def generate_square_subsequent_mask(sz, device):
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).T
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(
            mask == 1, float(0.0)
        )
        return mask

    def forward(self, src_nodes, tgt_seq):
        batch_size = src_nodes.size(0)
        cls_column = torch.full(
            (batch_size, 1), config.cls_token, dtype=torch.long, device=src_nodes.device
        )
        src_nodes_with_cls = torch.cat([cls_column, src_nodes], dim=1)

        src_emb = self.token_embedding(src_nodes_with_cls)
        src_emb = self.pos_encoder(src_emb)
        memory = self.encoder(src_emb)

        tgt_emb = self.token_embedding(tgt_seq)
        tgt_emb = self.pos_encoder(tgt_emb)
        tgt_mask = self.generate_square_subsequent_mask(
            tgt_seq.size(1), device=tgt_seq.device
        )

        dec_out = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        logits = self.output_fc(dec_out)
        return logits

    # ---------------- exposed helpers ----------------
    def encode(self, src_nodes):
        
        batch_size = src_nodes.size(0)
        cls_column = torch.full(
            (batch_size, 1), config.cls_token, dtype=torch.long, device=src_nodes.device
        )
        src_nodes_with_cls = torch.cat([cls_column, src_nodes], dim=1)
        src_emb = self.token_embedding(src_nodes_with_cls)
        src_emb = self.pos_encoder(src_emb)
        return self.encoder(src_emb)

    def decode_step(self, memory, partial_tgt_seq):
        tgt_emb = self.token_embedding(partial_tgt_seq)
        tgt_emb = self.pos_encoder(tgt_emb)
        mask = self.generate_square_subsequent_mask(
            partial_tgt_seq.size(1), partial_tgt_seq.device
        )
        dec = self.decoder(tgt_emb, memory, tgt_mask=mask)
        return self.output_fc(dec[:, -1, :])

    def get_cls_vector(self, memory):
        return memory[:, 0, :]


# ------------------------------------------------------------------
# TRAE wrapper
# ------------------------------------------------------------------
class TRAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TransformerAutoencoder()

    def forward(self, src_nodes, src_pos, tgt_seq):
        return self.model(src_nodes, tgt_seq)

    @property
    def encoder(self):
        return self.model.encode

    @property
    def decoder(self):
        return self.model

    def get_cls_summary(self, memory):
        return self.model.get_cls_vector(memory)


# ------------------------------------------------------------------
# Collate function
# ------------------------------------------------------------------
def collate_fn(batch):
    inputs, targets = [], []
    for expr in batch:
        flat = flatten_expr(expr)
        node_ids = [e["node_id"] for e in flat]
        tgt = [config.start_token] + node_ids + [config.end_token]
        inputs.append(node_ids)
        targets.append(tgt)

    max_src = max(len(x) for x in inputs)
    max_tgt = max(len(t) for t in targets)

    
    print(f"[collate] batch={len(batch)}  max_src={max_src}  max_tgt={max_tgt}")

    pad_src = [x + [config.pad_token] * (max_src - len(x)) for x in inputs]
    pad_tgt = [t + [config.pad_token] * (max_tgt - len(t)) for t in targets]

    return torch.tensor(pad_src, dtype=torch.long), torch.tensor(
        pad_tgt, dtype=torch.long
    )


# ------------------------------------------------------------------
# Evaluation / utility helpers 
# ------------------------------------------------------------------
def calculate_accuracy(preds, targets):
    exact_matches, correct_tokens, total_tokens = 0, 0, 0
    for pred, target in zip(preds, targets):
        clean_pred = [
            t
            for t in pred
            if t
            not in {
                config.start_token,
                config.end_token,
                config.pad_token,
                config.cls_token,
            }
        ]
        clean_target = [
            t
            for t in target
            if t
            not in {
                config.start_token,
                config.end_token,
                config.pad_token,
                config.cls_token,
            }
        ]
        clean_pred = clean_pred[:len(clean_target)]
        exact_matches += int(clean_pred == clean_target)

        min_len = min(len(clean_pred), len(clean_target))
        matches = sum(p == t for p, t in zip(clean_pred[:min_len], clean_target[:min_len]))
        correct_tokens += matches
        total_tokens += len(clean_target)

        if clean_pred != clean_target:
            print("not equal")
            print("clean_pred", clean_pred)
            print("clean_targ", clean_target)

    return {
        "exact": exact_matches / len(preds),
        "token": correct_tokens / total_tokens if total_tokens else 0,
    }

def get_expression_cls_embedding(expr, model):
    flat = flatten_expr(expr)
    node_ids = [e["node_id"] for e in flat]
    if len(node_ids) + 1 > config.max_seq_length:
        return None

    src_tensor = torch.tensor([node_ids], dtype=torch.long, device=device)
    memory = model.encoder(src_tensor)
    return model.get_cls_summary(memory)


def parameter_counts(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    return trainable_params, non_trainable_params


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )

    def forward(self, x):
        return self.net(x)


def contrastive_loss(z1, z2, label, temperature=0.5):
    # z1, z2: [batch, dim], label: [batch] (1=pos, 0=neg)
    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)
    sim = torch.sum(z1 * z2, dim=1) / temperature
    pos_loss = -nn.functional.logsigmoid(sim[label == 1]).mean() if (label == 1).any() else 0
    neg_loss = -nn.functional.logsigmoid(-sim[label == 0]).mean() if (label == 0).any() else 0
    return pos_loss + neg_loss

# returns a list of pairs of positive and negative pairs, 1 for positive and 0 for negative
def augment_expression(expr):
    
    """    
        A function to augment the expression by generating positive and negative pairs.
            expr: a string representing the expression
        Returns a list of tuples (expr_a, expr_b, label) where label is 1 for positive pairs and 0 for negative pairs. 
    """
    expr_pairs = []
    for i, c in enumerate(expr):
        if c == "+":
            prev_expr = expr
            expr = expr[:i] + "-" + expr[i + 1:]
            neg_pair = (prev_expr, expr, 0)  # negative pair
            pos_pair = (prev_expr, prev_expr, 1)
            expr_pairs.append(neg_pair)
            expr_pairs.append(pos_pair)

            expr = expr[:i] + "*" + expr[i + 1:] 
            neg_pair = (prev_expr, expr, 0)  # negative pair
            pos_pair = (prev_expr, prev_expr, 1)
            expr_pairs.append(neg_pair)
            expr_pairs.append(pos_pair)

        elif c == "-":
            prev_expr = expr
            expr = expr[:i] + "+" + expr[i + 1:]
            neg_pair = (prev_expr, expr, 0)  # negative pair
            pos_pair = (prev_expr, prev_expr, 1)
            expr_pairs.append(neg_pair)
            expr_pairs.append(pos_pair)

            expr = expr[:i] + "*" + expr[i + 1:] 
            neg_pair = (prev_expr, expr, 0)  # negative pair
            pos_pair = (prev_expr, prev_expr, 1)
            expr_pairs.append(neg_pair)
            expr_pairs.append(pos_pair)

        elif c == "*":
            prev_expr = expr
            expr = expr[:i] + "+" + expr[i + 1:]
            neg_pair = (prev_expr, expr, 0)  # negative pair
            pos_pair = (prev_expr, prev_expr, 1)
            expr_pairs.append(neg_pair)
            expr_pairs.append(pos_pair)

            expr = expr[:i] + "-" + expr[i + 1:] 
            neg_pair = (prev_expr, expr, 0)  # negative pair
            pos_pair = (prev_expr, prev_expr, 1)
            expr_pairs.append(neg_pair)
            expr_pairs.append(pos_pair)    
        
    return expr_pairs

# building the positive and negative contrastive pairs based on the expression strings
def build_contrastive_pairs(expr_strs):
    pairs = []
    # parse_sexpr
    for expr in expr_strs:
        expr_pairs = augment_expression(expr)

        # parsing the expressions of the generated pairs
        parsed_expr_pairs = [(parse_sexpr(a), parse_sexpr(b), label) for a, b, label in expr_pairs]
        pairs.extend(parsed_expr_pairs)
        
    return pairs


def demo():
    model = TRAE()
    model.eval()
    state_dict = torch.load("./fhe_rl/trained_models/model_Transformer_ddp_10399047_epoch_5000000.pth", map_location=device)
    new_sd = {k[len("module.") :] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    model.load_state_dict(new_sd)
    model.to(device)

    # Add projection head
    cls_dim = model.get_cls_summary(torch.zeros(1, 1, config.d_model, device=device)).shape[-1]
    projection_head = ProjectionHead(cls_dim, proj_dim=128).to(device)

    # Freeze encoder weights
    for param in model.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(projection_head.parameters(), lr=1e-3)

    # Load expressions from file
    with open("./fhe_rl/datasets/test.txt") as f:
        expr_strs = [line.strip() for line in f if line.strip()]
    
    pairs = build_contrastive_pairs(expr_strs)

    # storing the pairs in a file
    with open("./fhe_rl/datasets/contrastive_pairs.txt", "w") as f:
        for expr_a, expr_b, label in pairs:
            f.write(f"{expr_a} | {expr_b} | {label}\n")
    
    # Build pairs (more coverage)

    # Training loop
    for epoch in tqdm(range(10), desc="Training Epochs"):
        random.shuffle(pairs)
        losses = []
        
        # Add progress bar for the inner training loop
        with tqdm(pairs, desc=f"Epoch {epoch+1}/10", leave=False) as pbar:
            for expr_a, expr_b, label in pbar:
                cls_a = get_expression_cls_embedding(expr_a, model)
                cls_b = get_expression_cls_embedding(expr_b, model)
                
                if cls_a is None or cls_b is None:
                    continue
                cls_a = cls_a.to(device)
                cls_b = cls_b.to(device)
                z_a = projection_head(cls_a)
                z_b = projection_head(cls_b)
                lbl = torch.tensor([label], dtype=torch.long, device=device)
                loss = contrastive_loss(z_a, z_b, lbl)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                
                # Update progress bar with current loss
                if len(losses) > 0:
                    pbar.set_postfix({'loss': f'{np.mean(losses[-10:]):.6f}'})

        print(f"Epoch {epoch+1}: Contrastive loss = {np.mean(losses):.6f}")

    # save the projection head
    torch.save(projection_head.state_dict(), "./fhe_rl/trained_models/projection_head.pth")


def test():
    # loading the encoder model and printing the cosine similarity and the euclidean distance between the following two subexpressions:
    #  (Vec  (+ a b) (+ c d) (- f g) )
    # (Vec  (- a b) (- c d) (+ f g) )
    model = TRAE()
    model.eval()
    state_dict = torch.load("./fhe_rl/trained_models/model_Transformer_ddp_10399047_epoch_5000000.pth", map_location=device)

    new_sd = {k[len("module.") :] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    model.load_state_dict(new_sd)
    model.to(device)

    exp_a = parse_sexpr("(Vec  (+ a b) (+ c d) (- f g) )")
    exp_b = parse_sexpr("(Vec  (* a b) (* c d) (+ f g) )")
    #exp_b = parse_sexpr("(Vec  (- a b) (- c d) (+ f g) )")

    cls_a = get_expression_cls_embedding(exp_a, model)
    cls_b = get_expression_cls_embedding(exp_b, model)

    if cls_a is None or cls_b is None:
        print("Cannot encode them")
    
    # computing the cosine similarity between the embeddings
    print("Cosine similarity:", nn.functional.cosine_similarity(cls_a, cls_b).item())
    # computing the euclidean distance between the embeddings
    print("Euclidean distancce is", torch.norm(cls_a - cls_b).item())


def test_projection_head():
    model = TRAE()
    model.eval()
    state_dict = torch.load("./fhe_rl/trained_models/model_Transformer_ddp_10399047_epoch_5000000.pth", map_location=device)

    new_sd = {k[len("module.") :] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    model.load_state_dict(new_sd)
    model.to(device)

    # loading the projection head 
    # torch.save(projection_head.state_dict(), "./fhe_rl/trained_models/projection_head.pth")
    projection_head = ProjectionHead(model.get_cls_summary(torch.zeros(1, 1, config.d_model, device=device)).shape[-1], proj_dim=128).to(device)
    projection_head.load_state_dict(torch.load("./fhe_rl/trained_models/projection_head.pth", map_location=device))
    projection_head.eval()
    projection_head.to(device)

    exp_a = parse_sexpr("(Vec  (+ a b) (+ c d) (- f g) )")
    exp_b = parse_sexpr("(Vec  (* a b) (* c d) (+ f g) )")


    cls_a = get_expression_cls_embedding(exp_a, model)
    cls_b = get_expression_cls_embedding(exp_b, model)

    if cls_a is None or cls_b is None:
        print("Cannot encode them")
        return
    
    cls_a = cls_a.to(device)
    cls_b = cls_b.to(device)
    z_a = projection_head(cls_a)
    z_b = projection_head(cls_b)

    cosine_sim = nn.functional.cosine_similarity(z_a, z_b).item()
    print("Cosine similarity between test expressions:", cosine_sim)
    print("Embedding for expr_a:", z_a.cpu().detach().numpy())
    print("Embedding for expr_b:", z_b.cpu().detach().numpy())  



def test_augment_expression():
    expr = "(Vec  (+ a b) (+ c d) (- f g) )"
    pairs = build_contrastive_pairs([expr])
    return pairs

if __name__ == "__main__":
    # training the projection head
    demo()

    # testing the projection head
    test_projection_head()