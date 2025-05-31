import os
# Set environment variable for CUDA allocator to reduce fragmentation issues.
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch  
import torch.nn as nn  
import torch.optim as optim  
import torch.distributed as dist  
from torch.nn.parallel import DistributedDataParallel as DDP  
from torch.utils.data import DataLoader, DistributedSampler  
import math  
import random  
import sys, os  
import time  

# Append the pytrs directory to the path  
sys.path.append(os.path.abspath("./pytrs"))  
from veclang_gen import generate_multiple_expressions, MAX_INT_NUMBER  
from pytrs import (Op, VARIABLE_RANGE, CONST_OFFSET, PAREN_CLOSE, PAREN_OPEN, node_to_id, parse_sexpr)  

torch.set_float32_matmul_precision('high')  

def print(*args, **kwargs):  
    kwargs['flush'] = True  
    __builtins__.print(*args, **kwargs)  

# Subclass DDP to allow easier attribute access.  
class AEDDP(DDP):  
    def __getattr__(self, name):  
        try:  
            return super().__getattr__(name)  
        except AttributeError:  
            return getattr(self.module, name)  

############################################  
# DDP Setup and Device Assignment  
############################################  
ddp = int(os.environ.get('RANK', -1)) != -1  
deviceids = [0, 1, 2]  

if ddp:  
    assert torch.cuda.is_available(), "DDP requires CUDA"  
    dist.init_process_group(backend='nccl')  
    ddp_rank = int(os.environ['RANK'])  
    ddp_local_rank = int(os.environ['LOCAL_RANK'])  
    ddp_world_size = int(os.environ['WORLD_SIZE'])  
    device = f'cuda:{deviceids[ddp_rank]}'  
    torch.cuda.set_device(device)  
    master_process = ddp_rank == 0  
else:  
    master_process = True  
    ddp_rank = 0  
    ddp_world_size = 1  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

if master_process:    
    print("Using DDP with world size:", ddp_world_size)  
    print("Assigned device:", device)  

############################################  
# 1) Configuration and Token Setup  
############################################  
class Config:  
    # we no longer need max_seq_length for long sequences  
    vocab_size     = CONST_OFFSET + MAX_INT_NUMBER + 3  # + START, END, PAD  
    start_token    = CONST_OFFSET + MAX_INT_NUMBER  
    end_token      = CONST_OFFSET + MAX_INT_NUMBER + 1  
    pad_token      = CONST_OFFSET + MAX_INT_NUMBER + 2  
    max_depth      = 15
    max_arity      = 32  # maximum number of children for a node
    # dimensions  
    d_model        = 256  
    decoder_hidden = 256  
    decoder_layers = 2  

    # Data parameters (same as before)  
    total_samples = 5000000  
    train_ratio   = 1  
    valid_ratio   = 0  
    test_ratio    = 0  
    batch_size     = 16  
    learning_rate  = 3e-4  
    epochs         = 50  

config = Config()  

############################################  
# 2) Expression Processing (for targets only)  
############################################  
def dfs_traverse(expr, node_list=None):
    """
    Depth-first walk, emitting:
      - PAREN_OPEN / PAREN_CLOSE (they're ints)
      - Op objects or leaf symbols
    """
    if node_list is None:
        node_list = []
    if isinstance(expr, Op):
        node_list.append(PAREN_OPEN)      # int
        node_list.append(expr)            # Op
        for child in expr.args:
            dfs_traverse(child, node_list)
        node_list.append(PAREN_CLOSE)     # int
    else:
        node_list.append(expr)            # leaf (variable/constant)
    return node_list

def flatten_expr(expr):
    """
    Turn a parsed `Op` tree into a list of integer token‐IDs.
    Skips re‐mapping any ints (the PAREN_OPEN/CLOSE).
    """
    node_list = dfs_traverse(expr)
    varmap, intmap = {}, {}
    next_var_id, next_int_id = VARIABLE_RANGE[0], CONST_OFFSET

    ids = []
    for node in node_list:
        if isinstance(node, int):
            # already a parenthesis token
            ids.append(node)
        else:
            nid, next_var_id, next_int_id, _ = node_to_id(
                node, varmap, intmap, next_var_id, next_int_id
            )
            ids.append(nid)
    return ids

# 4) New collate_fn that keeps src as Expr trees
def collate_fn(batch):
    """
    batch: list of parsed expressions (Op trees)
    returns:
      - src_exprs: list[Op]
      - tgt_tensor: LongTensor of shape (B, T)
    """
    src_exprs = batch[:]  # pass full trees straight to your encoder

    # build target ID sequences with START/END tokens
    tgt_seqs = []
    for expr in batch:
        flat_ids = flatten_expr(expr)
        seq = [config.start_token] + flat_ids + [config.end_token]
        tgt_seqs.append(seq)

    # pad to max length in batch
    max_len = max(len(s) for s in tgt_seqs)
    padded = [
        s + [config.pad_token] * (max_len - len(s))
        for s in tgt_seqs
    ]
    tgt_tensor = torch.tensor(padded, dtype=torch.long)

    return src_exprs, tgt_tensor

############################################  
# 3) Tree‐Structured Encoder + GRU Decoder  
############################################  
############################################  
# 3) Tree‐Structured Encoder + GRU Decoder  
############################################  

class OpCombiner(nn.Module):  
    def __init__(self):  
        super().__init__()  
        # token embedding  
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)  
        # two positional embeddings: one for depth, one for child-index (0 or 1)  
        self.depth_emb = nn.Embedding(config.max_depth, config.d_model)  
        self.idx_emb   = nn.Embedding(2,       config.d_model)  
        # combine op + left + right + depth(left) + idx(left) + depth(right) + idx(right)  
        self.fc = nn.Sequential(  
            nn.Linear(config.d_model * 7, config.d_model),  
            nn.ReLU(),  
            nn.LayerNorm(config.d_model),  
        )  

    def forward(self, nid, left, right, dl, il, dr, ir):  
        e_op = self.token_emb(nid)  
        e_l  = left  
        e_r  = right  
        e_dl = self.depth_emb(dl)  
        e_il = self.idx_emb(il)  
        e_dr = self.depth_emb(dr)  
        e_ir = self.idx_emb(ir)  
        x = torch.cat([e_op, e_l, e_r, e_dl, e_il, e_dr, e_ir], dim=-1)  
        return self.fc(x)  


class VecCombiner(nn.Module):  
    def __init__(self):  
        super().__init__()  
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)  
        self.depth_emb = nn.Embedding(config.max_depth, config.d_model)  
        # index embedding now covers up to config.max_arity slots  
        self.idx_emb   = nn.Embedding(config.max_arity, config.d_model)  

        # we'll allocate: 1×op + for each slot (emb + depth + idx) => total dims = d_model*(1 + 3*config.max_arity)  
        self.fc = nn.Sequential(  
            nn.Linear(config.d_model * (1 + 3 * config.max_arity), config.d_model),  
            nn.ReLU(),  
            nn.LayerNorm(config.d_model),  
        )  

    def forward(self, nid, child_embs, child_depths):  
        B = nid.size(0)  
        # pad embeddings, depths to config.max_arity  
        pad_emb = torch.zeros(1, config.d_model, device=nid.device)  
        pad_dep = torch.zeros(1, dtype=torch.long, device=nid.device)  
        embs = child_embs + [pad_emb] * (config.max_arity - len(child_embs))  
        deps = child_depths + [pad_dep] * (config.max_arity - len(child_depths))  

        e_op = self.token_emb(nid)  
        parts = [e_op]  
        # for each slot i, include emb, depth_emb, idx_emb(i)  
        for i in range(config.max_arity):  
            parts.append(embs[i])                            # child embedding  
            parts.append(self.depth_emb(deps[i]))            # depth at which child appeared  
            parts.append(self.idx_emb(torch.tensor(i, device=nid.device)))  # position i in Vec  
        x = torch.cat(parts, dim=-1)  
        return self.fc(x)  


class TreeStructuredEncoder(nn.Module):  
    def __init__(self):  
        super().__init__()  
        self.op_comb  = OpCombiner()  
        self.vec_comb = VecCombiner()  

    def forward(self, expr):  
        emb, _, _ = self._encode(expr, depth=0)  
        return emb  

    def _encode(self, node, depth):  
        # leaf  
        if not isinstance(node, Op):  
            nid, _, _, _ = node_to_id(node, {}, {}, VARIABLE_RANGE[0], CONST_OFFSET)  
            e_tok = self.op_comb.token_emb(torch.tensor([nid], device=device))  
            e_pos = self.op_comb.depth_emb(torch.tensor([depth], device=device))  
            return e_tok + e_pos, None, None  

        # Vec (n‐ary)  
        if node.op == 'Vec':  
            child_embs, child_depths = [], []  
            for idx, c in enumerate(node.args):  
                e, _, _ = self._encode(c, depth + 1)  
                child_embs.append(e)  
                child_depths.append(torch.tensor([depth+1], device=device))  
            nid, _, _, _ = node_to_id(node, {}, {}, VARIABLE_RANGE[0], CONST_OFFSET)  
            return self.vec_comb(torch.tensor([nid], device=device),  
                                 child_embs, child_depths), None, None  

        # binary/unary operator  
        left, _, _ = self._encode(node.args[0], depth + 1)  
        idx_l = torch.tensor([0], device=device)  
        dl    = torch.tensor([depth+1], device=device)  

        if len(node.args) == 2:  
            right, _, _ = self._encode(node.args[1], depth + 1)  
            idx_r = torch.tensor([1], device=device)  
            dr    = torch.tensor([depth+1], device=device)  
        else:  
            right = torch.zeros_like(left)  
            idx_r = torch.tensor([0], device=device)  
            dr    = torch.tensor([0], device=device)  

        nid, _, _, _ = node_to_id(node, {}, {}, VARIABLE_RANGE[0], CONST_OFFSET)  
        return self.op_comb(torch.tensor([nid], device=device),  
                             left, right, dl, idx_l, dr, idx_r), None, None  

class GRUDecoder(nn.Module):  
    def __init__(self):  
        super().__init__()  
        self.embed = nn.Embedding(config.vocab_size, config.d_model)  
        self.gru   = nn.GRU(config.d_model, config.decoder_hidden,  
                            num_layers=config.decoder_layers, batch_first=True)  
        self.fc    = nn.Linear(config.decoder_hidden, config.vocab_size)  

    def forward(self, hidden, tgt_seq):  
        emb = self.embed(tgt_seq)            # (B,T,d)  
        out, _ = self.gru(emb, hidden)       # (B,T,H)  
        return self.fc(out)                  # (B,T,V)  

class TreeAutoencoder(nn.Module):  
    def __init__(self):  
        super().__init__()  
        self.encoder     = TreeStructuredEncoder()  
        self.hidden_proj = nn.Linear(config.d_model, config.decoder_hidden*config.decoder_layers)  
        self.decoder     = GRUDecoder()  

    def forward(self, src_exprs, tgt_seq):  
        # src_exprs: list of parse_sexpr objects  
        embs = [self.encoder(e) for e in src_exprs]  
        H = torch.stack(embs, dim=0)                   # (B,d)  
        h0 = self.hidden_proj(H)                       # (B, L*H)  
        B = H.size(0)  
        h0 = h0.view(B, config.decoder_layers, config.decoder_hidden)  
        h0 = h0.permute(1,0,2).contiguous()            # (L,B,H)  
        return self.decoder(h0, tgt_seq)               # (B,T,V)  


############################################  
# 5) Training Function (Validation Removed)  
############################################  
def train_autoenc(model, train_dataset):  
    train_sampler = DistributedSampler(train_dataset) if ddp else None  
    train_loader  = DataLoader(train_dataset, batch_size=config.batch_size,  
                        shuffle=(train_sampler is None), sampler=train_sampler,  
                        collate_fn=collate_fn, num_workers=1, pin_memory=True)  
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)  
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)  
    loss_fn = nn.CrossEntropyLoss(ignore_index=config.pad_token)  

    print(f"[Rank {ddp_rank}] Starting training for {config.epochs} epochs.")  
    for epoch in range(config.epochs):  
        print(f"[Rank {ddp_rank}] Epoch {epoch+1} start")  
        epoch_start = time.time()  
        model.train()  
        if train_sampler: 
            train_sampler.set_epoch(epoch)  
        total_loss = 0.0
        num_batches = 0  

        for exprs, tgt in train_loader:
            tgt = tgt.to(device)
            logits = model(exprs, tgt[:,:-1])  
            loss = loss_fn(logits.view(-1, config.vocab_size), tgt[:,1:].reshape(-1))  
            optimizer.zero_grad()  
            loss.backward()  
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  
            optimizer.step()  

            total_loss += loss.item()  
            num_batches += 1  

        epoch_end = time.time()  
        if master_process:  
            print(f"[Rank {ddp_rank}] Epoch {epoch+1} done in {epoch_end-epoch_start:.1f}s, loss={total_loss/num_batches:.4f}")  
        scheduler.step()  
        if ddp: dist.barrier()  

############################################  
# 6) Test Function (unchanged)  
############################################  
def test_autoenc(model, dataset, return_samples=False, stdout=False):  
    loader = DataLoader(dataset, batch_size=config.batch_size,  
                        shuffle=False, collate_fn=collate_fn)  
    loss_fn = nn.CrossEntropyLoss(ignore_index=config.pad_token)  
    model.eval()  
    total_loss = 0.0  

    with torch.no_grad():  
        for exprs, tgt in loader:  
            logits = model(exprs, tgt[:,:-1])  
            total_loss += loss_fn(logits.view(-1, config.vocab_size), tgt[:,1:].reshape(-1)).item()  

    print(f"Test Loss: {total_loss/len(loader):.4f}")  
    return {}, ""  # adjust if you need accuracy  

############################################  
# 7) Demo Run  
############################################  
def demo_run():  
    if sys.argv[1].lower()=="train":  
        print("Loading dataset...")  
        with open("/scratch/ad7786/chehab-vectorization-rl/pretraining/data50.txt") as f:  
            all_expressions = [x.strip() for x in f]  
            
        random.seed(52)
        random.shuffle(all_expressions)  
        train_dataset = [parse_sexpr(e) for e in all_expressions]  

        model = TreeAutoencoder().to(device)  
        if ddp: 
            model = AEDDP(model, device_ids=[deviceids[ddp_rank]])  
        train_autoenc(model, train_dataset)  
        if ddp: 
            dist.destroy_process_group()  

    else:  # test  
        model = TreeAutoencoder().to(device)  
        model.load_state_dict(torch.load(sys.argv[2], map_location=device))  
        test_exprs = [parse_sexpr(l.strip()) for l in open("llm_collapsed.txt")]  
        test_autoenc(model, test_exprs, stdout=True)  

if __name__=="__main__":  
    demo_run()  
