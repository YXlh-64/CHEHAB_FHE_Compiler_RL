###############################################################################
# hierarchy_autoenc.py – hierarchical encoder + edge-attention decoder
###############################################################################
import os, sys, random, math, torch
import torch.nn as nn, torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

# ─────────────────── 0. Environment & DDP ──────────────────────────────────
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    dist.init_process_group('nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device  = torch.device('cuda', local_rank)
    master  = local_rank == 0
else:
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    master, local_rank = True, 0
deviceids = [local_rank]

# ─────────────────── 1. Project imports ────────────────────────────────────
sys.path.append(os.path.abspath('./pytrs'))
from pytrs import (Op, VARIABLE_RANGE, CONST_OFFSET, node_to_id, parse_sexpr)
from pretraining.veclang_gen import MAX_INT_NUMBER

# ─────────────────── 2. Hyper-parameters ───────────────────────────────────
class Config:
    d_model = 256
    heads   = 4
    dropout = 0.1
    batch   = 32
    lr      = 3e-4
    epochs  = 50
    max_len = 1500
    vocab   = CONST_OFFSET + MAX_INT_NUMBER + 3
    start   = CONST_OFFSET + MAX_INT_NUMBER
    end     = CONST_OFFSET + MAX_INT_NUMBER + 1
    pad     = CONST_OFFSET + MAX_INT_NUMBER + 2
C = Config()

# ─────────────────── 3. Tree helpers ───────────────────────────────────────
def to_tree(expr):
    nodes, varmap, intmap = [], {}, {}
    nv, ni = VARIABLE_RANGE[0], CONST_OFFSET
    def rec(node, parent=None):
        nonlocal nv, ni
        idx = len(nodes)
        tok, nv, ni, _ = node_to_id(node, varmap, intmap, nv, ni)
        nodes.append({'tok': tok, 'children': []})
        if parent is not None:
            nodes[parent]['children'].append(idx)
        if isinstance(node, Op):
            for ch in node.args: rec(ch, idx)
    rec(expr)
    return {'nodes': nodes, 'root': 0}

# ─────────────────── 4. Encoder – parent/child edge-attention ──────────────
class ComposerAttn(nn.Module):
    """
    parent_vec (query) attends to its children (keys/values + position-emb).
    Residual-Dropout-LayerNorm included.
    """
    def __init__(self, d, heads=4, max_children=32):
        super().__init__()
        self.h, self.dk = heads, d // heads
        assert d % heads == 0
        self.q_proj = nn.Linear(d, d, bias=False)
        self.kv_proj= nn.Linear(d, 2*d, bias=False)
        self.pos    = nn.Embedding(max_children, d)
        self.out    = nn.Sequential(nn.Linear(d, d),
                                    nn.Dropout(C.dropout))
        self.ln     = nn.LayerNorm(d)

    def forward(self, parent_vec, child_pairs):          # child_pairs = [(vec, idx)]
        if not child_pairs:                              # leaf
            return parent_vec

        vecs, idxs = zip(*child_pairs)
        vecs = [v + self.pos(torch.tensor(i, device=v.device))
                for v, i in zip(vecs, idxs)]             # (k,d)
        kv   = torch.stack(vecs)                         # (k,d)
        k, v = self.kv_proj(kv).chunk(2, dim=-1)         # each (k,d)

        k = k.view(len(kv), self.h, self.dk)             # (k,H,d_k)
        v = v.view(len(kv), self.h, self.dk)
        q = self.q_proj(parent_vec).view(self.h, self.dk)# (H,d_k)

        scores = (k * q).sum(-1) * self.dk**-0.5         # (k,H)
        att    = scores.softmax(0)                       # (k,H)
        ctx    = (att.unsqueeze(-1) * v).sum(0).reshape(-1)  # (d,)
        return self.ln(parent_vec + self.out(ctx))       # residual & LN

class HierarchicalEncoder(nn.Module):
    def __init__(self, vocab, d):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.comp  = ComposerAttn(d, C.heads)

    def embed_tree(self, tree):
        cache = {}                                       # fresh cache per tree
        def rec(idx):
            if idx in cache: return cache[idx]
            node = tree['nodes'][idx]
            parent_vec = self.embed.weight[node['tok']]
            if node['children']:
                children = [(rec(c), j) for j, c in enumerate(node['children'])]
                vec = self.comp(parent_vec, children)
            else:
                vec = parent_vec
            cache[idx] = vec
            return vec
        return rec(tree['root'])

    def forward(self, trees):
        return torch.stack([self.embed_tree(t) for t in trees])

# ─────────────────── 5. Decoder + full model ───────────────────────────────
def causal_mask(n, dev):
    return torch.triu(torch.full((n, n), float('-inf'), device=dev), 1)

class HierarchicalAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = HierarchicalEncoder(C.vocab, C.d_model)
        dec_layer = nn.TransformerDecoderLayer(C.d_model, 8,
                                               4*C.d_model, batch_first=True,
                                               dropout=C.dropout)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=4)
        self.tok  = nn.Embedding(C.vocab, C.d_model)
        self.proj = nn.Linear(C.d_model, C.vocab)

    def forward(self, trees, tgt_in):
        mem = self.encoder(trees)                        # (B,d)
        out = self.decoder(self.tok(tgt_in), mem.unsqueeze(1),
                           tgt_mask=causal_mask(tgt_in.size(1), tgt_in.device))
        return self.proj(out)

    @torch.no_grad()
    def decode_step(self, mem, part):
        out = self.decoder(self.tok(part), mem.unsqueeze(1),
                           tgt_mask=causal_mask(part.size(1), part.device))
        return self.proj(out[:, -1])

# ─────────────────── 6. Data pipeline ──────────────────────────────────────
def collate(batch):
    trees = [to_tree(e) for e in batch]
    tgt   = [[C.start] + [n['tok'] for n in t['nodes']] + [C.end] for t in trees]
    T     = max(len(s) for s in tgt)
    tgt   = [s + [C.pad]*(T-len(s)) for s in tgt]
    return trees, torch.tensor(tgt, dtype=torch.long)

# ─────────────────── 7. Training + validation ──────────────────────────────
def run_epoch(model, loader, loss_fn, train=True, opt=None):
    model.train() if train else model.eval()
    total=n=0
    with torch.set_grad_enabled(train):
        for trees, tgt in loader:
            tgt = tgt.to(device)
            loss = loss_fn(model(trees, tgt[:, :-1]).reshape(-1, C.vocab),
                           tgt[:, 1:].reshape(-1))
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad()
            total += loss.item(); n += 1
    return total/n

def train(model, train_set, val_set):
    tr_ld = DataLoader(train_set, batch_size=C.batch, collate_fn=collate,
                       sampler=DistributedSampler(train_set) if ddp else None,
                       shuffle=not ddp)
    va_ld = DataLoader(val_set,   batch_size=C.batch, collate_fn=collate,
                       sampler=DistributedSampler(val_set, shuffle=False) if ddp else None)

    loss_fn = nn.CrossEntropyLoss(ignore_index=C.pad)
    opt     = optim.AdamW(model.parameters(), lr=C.lr, betas=(0.9,0.98))
    sched   = optim.lr_scheduler.CosineAnnealingLR(opt, C.epochs)
    best    = float('inf')

    job = os.environ.get('SLURM_JOB_ID', 'nojid')
    os.makedirs('pretraining/saved_models', exist_ok=True)

    for ep in range(1, C.epochs+1):
        tr = run_epoch(model, tr_ld, loss_fn, True,  opt)
        vl = run_epoch(model, va_ld, loss_fn, False)
        sched.step()
        if master:
            print(f"ep {ep:02d}  train {tr:.4f}  val {vl:.4f}")
            if vl < best - 1e-4:
                best = vl
                fname = f'pretraining/saved_models/hier_{job}_best.pth'
                torch.save(model.module.state_dict() if isinstance(model, DDP)
                           else model.state_dict(), fname)
                print("  ↳ saved new best")

# ─────────────────── 8. Embedding utility ─────────────────────────────────
@torch.no_grad()
def get_expression_embedding(expr_str: str,
                              model: "HierarchicalAutoencoder",
                              device: torch.device = device):
    tree = to_tree(parse_sexpr(expr_str))
    return model.encoder([tree]).squeeze(0).cpu()        # (d_model,)

# ─────────────────── 9. CLI ───────────────────────────────────────────────
def main():
    mode = sys.argv[1].lower()

    if mode == 'train':
        with open('pretraining/dataset_balanced_8_2_1000000.txt') as f:
            all_expr = [parse_sexpr(l.strip()) for l in f]

        random.shuffle(all_expr)
        split = int(0.9*len(all_expr))
        train_set, val_set = all_expr[:split], all_expr[split:]
        if master: print("Train:", len(train_set), " Val:", len(val_set))

        model = HierarchicalAutoencoder().to(device)
        if ddp and dist.get_world_size() > 1:
            model = DDP(model, device_ids=deviceids,
                        find_unused_parameters=True)

        train(model, train_set, val_set)
        if ddp: dist.destroy_process_group()

    elif mode == 'test':
        ckpt  = sys.argv[2]
        model = HierarchicalAutoencoder().to(device)
        sd    = torch.load(ckpt, map_location=device)
        model.load_state_dict({k.replace('module.',''):v for k,v in sd.items()})
        model.eval()
        print(get_expression_embedding("(VecMinus (Vec a1629) (Vec 103))", model))

if __name__ == '__main__':
    main()
