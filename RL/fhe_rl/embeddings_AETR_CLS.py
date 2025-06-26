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
import numpy as np  

# Append the pytrs directory to the path  
sys.path.append(os.path.abspath("./pytrs"))  
from veclang_gen import generate_multiple_expressions, MAX_INT_NUMBER  
from pytrs import (Op, VARIABLE_RANGE, CONST_OFFSET, PAREN_CLOSE, PAREN_OPEN, node_to_id, parse_sexpr)  

torch.set_float32_matmul_precision('high')  

# def print(*args, **kwargs):  
#     kwargs['flush'] = True  
#     __builtins__.print(*args, **kwargs)  

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
    max_gen_length = 25122  
    # max_depth = 10  
    # vector_size = 9  

    vocab_size = CONST_OFFSET + MAX_INT_NUMBER + 2 + 1 + 1  
    start_token = CONST_OFFSET + MAX_INT_NUMBER   
    end_token = CONST_OFFSET + MAX_INT_NUMBER + 1  
    pad_token = CONST_OFFSET + MAX_INT_NUMBER + 2  

    # Add a new CLS token  
    cls_token = vocab_size  
    vocab_size += 1  # increment the vocab to include CLS  

    d_model = 256  
    num_heads = 8  
    num_encoder_layers = 4  
    num_decoder_layers = 4  
    dim_feedforward = 512   
    transformer_dropout = 0.2  
    max_seq_length = 25200    

    # Adjust batch size if necessary to reduce memory footprint
    batch_size = 16  # You might try reducing this if OOM persists
    learning_rate = 3e-4  
    epochs = 50
    dropout_rate = 0.3  

    # Data parameters  
    total_samples = 5000000  
    train_ratio = 1 
    valid_ratio = 0 
    test_ratio = 0  

config = Config()  

############################################  
# 2) Expression Processing Functions  
############################################  
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
        if node_or_paren == PAREN_OPEN or node_or_paren == PAREN_CLOSE:  
            nid = node_or_paren  
        else:  
            nid, next_var_id, next_int_id, _ = node_to_id(  
                node_or_paren, varmap, intmap, next_var_id, next_int_id  
            )  
        results.append({"node_id": nid})  
    return results  

############################################  
# 3) Positional Encoding Module  
############################################  
class PositionalEncoding(nn.Module):  
    def __init__(self, d_model, max_len=1024):  
        super().__init__()  
        self.pos_embedding = nn.Embedding(max_len, d_model)  

    def forward(self, x, positions=None):  
        batch_size, seq_len, d_model = x.shape  
        if positions is None:  
            positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).repeat(batch_size, 1)  
        pos_emb = self.pos_embedding(positions)  
        return x + pos_emb  

############################################  
# 4) Transformer Autoencoder Architecture  
############################################  
class TransformerAutoencoder(nn.Module):  
    def __init__(self):  
        super().__init__()  
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)  
        self.pos_encoder = PositionalEncoding(config.d_model, max_len=config.max_seq_length)  
        encoder_layer = nn.TransformerEncoderLayer(  
            d_model=config.d_model,  
            nhead=config.num_heads,  
            dim_feedforward=config.dim_feedforward,  
            dropout=config.transformer_dropout,  
            batch_first=True  
        )  
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_encoder_layers)  
        decoder_layer = nn.TransformerDecoderLayer(  
            d_model=config.d_model,  
            nhead=config.num_heads,  
            dim_feedforward=config.dim_feedforward,  
            dropout=config.transformer_dropout,  
            batch_first=True  
        )  
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.num_decoder_layers)  
        self.output_fc = nn.Linear(config.d_model, config.vocab_size)  

    @staticmethod  
    def generate_square_subsequent_mask(sz, device):  
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)  
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))  
        return mask  

    def forward(self, src_nodes, tgt_seq):  
        batch_size = src_nodes.size(0)  
        cls_column = torch.full((batch_size, 1), config.cls_token, dtype=torch.long, device=src_nodes.device)  
        src_nodes_with_cls = torch.cat([cls_column, src_nodes], dim=1)  
        src_emb = self.token_embedding(src_nodes_with_cls)  
        src_emb = self.pos_encoder(src_emb)  
        memory = self.encoder(src_emb)  
        tgt_emb = self.token_embedding(tgt_seq)  
        tgt_emb = self.pos_encoder(tgt_emb)  
        tgt_len = tgt_seq.size(1)  
        tgt_mask = self.generate_square_subsequent_mask(tgt_len, device=tgt_seq.device)  
        decoder_out = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)  
        logits = self.output_fc(decoder_out)  
        return logits  

    def encode(self, src_nodes):  
        batch_size = src_nodes.size(0)  
        cls_column = torch.full((batch_size, 1), config.cls_token, dtype=torch.long, device=src_nodes.device)  
        src_nodes_with_cls = torch.cat([cls_column, src_nodes], dim=1)  
        src_emb = self.token_embedding(src_nodes_with_cls)  
        src_emb = self.pos_encoder(src_emb)  
        memory = self.encoder(src_emb)  
        return memory  

    def decode_step(self, memory, partial_tgt_seq):  
        tgt_emb = self.token_embedding(partial_tgt_seq)  
        tgt_emb = self.pos_encoder(tgt_emb)  
        cur_len = partial_tgt_seq.size(1)  
        mask = self.generate_square_subsequent_mask(cur_len, partial_tgt_seq.device)  
        dec_out = self.decoder(tgt_emb, memory, tgt_mask=mask)  
        next_logits = self.output_fc(dec_out[:, -1, :])  
        return next_logits  

    def get_cls_vector(self, memory):  
        return memory[:, 0, :]  

############################################  
# 5) TreeAutoencoder Wrapper  
############################################  
class TreeAutoencoder(nn.Module):  
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

############################################  
# 6) Collate Function for DataLoader  
############################################  
def collate_fn(batch):  
    inputs = []  
    targets = []  
    for expr in batch:  
        flat = flatten_expr(expr)  
        node_ids = [entry["node_id"] for entry in flat]  
        tgt = [config.start_token] + node_ids + [config.end_token]  
        inputs.append(node_ids)  
        targets.append(tgt)  
    max_src_len = max(len(x) for x in inputs)  
    max_tgt_len = max(len(t) for t in targets)  
    padded_src = [x + [config.pad_token] * (max_src_len - len(x)) for x in inputs]  
    padded_tgt = [t + [config.pad_token] * (max_tgt_len - len(t)) for t in targets]  
    src_tensor = torch.tensor(padded_src, dtype=torch.long)  
    tgt_tensor = torch.tensor(padded_tgt, dtype=torch.long)  
    return src_tensor, tgt_tensor  

############################################  
# 7) Training Function (Validation Removed)  
############################################  
def train_autoenc(model, train_dataset):  
    # If using gradient accumulation, set the accumulation steps here:
    accumulation_steps = 1  # Increase if you want to simulate a larger batch size

    train_sampler = DistributedSampler(train_dataset) if ddp else None  
    train_loader = DataLoader(  
        train_dataset,  
        batch_size=config.batch_size,  
        shuffle=(train_sampler is None),  
        sampler=train_sampler,  
        collate_fn=collate_fn,  
        num_workers=1,    
        pin_memory=True  
    )  

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)  
    # Using a StepLR scheduler to decrease learning rate every epoch  
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
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (src_nodes, tgt_seq) in enumerate(train_loader):  
            src_nodes = src_nodes.to(device)  
            tgt_seq = tgt_seq.to(device)  
            # Note: forward call uses src_nodes and targets with shifted inputs
            logits = model(src_nodes, None, tgt_seq[:, :-1])  
            loss = loss_fn(logits.reshape(-1, config.vocab_size),  
                             tgt_seq[:, 1:].contiguous().view(-1))  
            loss = loss / accumulation_steps  # Normalize loss if using accumulation  
            loss.backward()  

            if (batch_idx + 1) % accumulation_steps == 0:  
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  
                optimizer.step()  
                optimizer.zero_grad(set_to_none=True)  

            total_loss += loss.item()  
            num_batches += 1  

        epoch_end = time.time()  
        epoch_duration = epoch_end - epoch_start  
        if ddp:  
            duration_tensor = torch.tensor(epoch_duration, device=device)  
            dist.all_reduce(duration_tensor, op=dist.ReduceOp.MAX)  
            epoch_duration = duration_tensor.item()  
        avg_train_loss = total_loss / num_batches  
        if master_process:  
            print(f"[Rank {ddp_rank}] Epoch {epoch+1} completed in {epoch_duration:.2f} seconds. Average Loss: {avg_train_loss:.4f}\n")  

        # Update the learning rate scheduler after each epoch  
        scheduler.step()  
        print(f"[Rank {ddp_rank}] Learning rate after epoch {epoch+1}: {optimizer.param_groups[0]['lr']:.6f}")  

        # Clear cache at the end of epoch to free unused memory
        torch.cuda.empty_cache()

        # Save the model at the end of the epoch  
        if master_process:  
            job_id = os.environ.get('SLURM_JOB_ID', "jobid")  
            model_name = f'model_Transformer_ddp_{job_id}_epoch_{config.total_samples}.pth'  
            os.makedirs("saved_models", exist_ok=True)  
            torch.save(model.state_dict(), os.path.join("saved_models", model_name))  
            print(f"[Rank {ddp_rank}] Model saved: {model_name}")  

        if ddp:  
            dist.barrier()  

############################################  
# 8) Test Function (unchanged, aside from potential memory cleanup if needed)  
############################################  
def test_autoenc(model, dataset, return_samples=False, stdout=False):
    model.eval()
    all_preds = []
    all_targets = []
    sample_examples = []

    with torch.no_grad():
        for i in range(0, len(dataset), config.batch_size):
            batch = dataset[i:i+config.batch_size]
            if not batch:
                continue
            orig_exprs = []
            inputs, targets = [], []
            for expr in batch:
                orig_exprs.append(expr)
                flat = flatten_expr(expr)
                node_ids = [f["node_id"] for f in flat]
                tgt_seq = [config.start_token] + node_ids + [config.end_token]
                inputs.append(node_ids)
                targets.append(tgt_seq)

            max_src_len = max(len(x) for x in inputs)
            max_tgt_len = max(len(t) for t in targets)
            padded_src = []
            padded_tgt = []
            for src_seq, tgt_seq in zip(inputs, targets):
                src_seq_pad = src_seq + [config.pad_token]*(max_src_len - len(src_seq))
                tgt_seq_pad = tgt_seq + [config.pad_token]*(max_tgt_len - len(tgt_seq))
                padded_src.append(src_seq_pad)
                padded_tgt.append(tgt_seq_pad)

            src_nodes = torch.tensor(padded_src, dtype=torch.long, device=device)
            memory = model.encoder(src_nodes)  # shape (batch, src_len+1, d_model)
            
            batch_preds = []
            for j in range(len(memory)):
                partial_seq = torch.tensor([[config.start_token]], dtype=torch.long, device=device)
                pred_tokens = []
                for _ in range(config.max_gen_length):
                    next_logits = model.decoder.decode_step(memory[j].unsqueeze(0), partial_seq)
                    next_token = next_logits.argmax(dim=-1).item()
                    pred_tokens.append(next_token)
                    if next_token == config.end_token:
                        break
                    
                    partial_seq = torch.cat([partial_seq, torch.tensor([[next_token]], device=device)], dim=1)

                batch_preds.append(pred_tokens)

            all_preds.extend(batch_preds)
            all_targets.extend(padded_tgt)

            if stdout:
                for orig_expr, target, pred in zip(orig_exprs, padded_tgt, batch_preds):
                    print("Expression:", orig_expr)
                    print("Target:    ", target[1:])
                    print("Predicted: ", pred)
                    print("")

            if return_samples and len(sample_examples) < 5:
                for idx in range(min(2, len(batch_preds))):
                    sample_examples.append({
                        "target": padded_tgt[idx],
                        "prediction": batch_preds[idx]
                    })

    sample_text = ""
    if sample_examples:
        for ex in sample_examples:
            sample_text += f"Target: {ex['target']}\nPred : {ex['prediction']}\n\n"

    return calculate_accuracy(all_preds, all_targets), sample_text

def calculate_accuracy(preds, targets):
    exact_matches = 0
    correct_tokens = 0
    total_tokens = 0
    
    for pred, target in zip(preds, targets):
        clean_pred = [t for t in pred if t not in {config.start_token, config.end_token, config.pad_token, config.cls_token}]
        clean_target = [t for t in target if t not in {config.start_token, config.end_token, config.pad_token, config.cls_token}]
        a = min(len(clean_pred),len(clean_target))
        exact_matches += int(clean_pred == clean_target)
        if not clean_pred == clean_target : 
            print("not equal")
            print("clean_pred", clean_pred)
            print("clean_targ", clean_target)
        min_len = min(len(clean_pred), len(clean_target))
        if min_len == 0:
            continue
        matches = sum(p == t for p, t in zip(clean_pred[:min_len], clean_target[:min_len]))
        correct_tokens += matches
        total_tokens += len(clean_target)
    
    return {
        'exact': exact_matches / len(preds),
        'token': correct_tokens / total_tokens if total_tokens else 0
    }
    

def get_expression_cls_embedding(expr, model):
    """
    Compute and return the CLS embedding for an expression using the provided model.
    
    Parameters:
        expr: The expression to encode (parsed using `parse_sexpr`).
        model: Instance of TreeAutoencoder.
    
    Returns:
        The CLS embedding tensor extracted from the encoder memory.
    """
    # Flatten the expression into a sequence of node dictionaries.
    flat = flatten_expr(expr)
    # Extract the node IDs into a list.
    node_ids = [entry["node_id"] for entry in flat]
    if len(node_ids) + 1 > config.max_seq_length:
        return None
    # Create a source tensor with a batch dimension.
    # (The model expects inputs of shape [batch_size, sequence_length])
    src_tensor = torch.tensor([node_ids], dtype=torch.long, device=device)
    
    # Run the encoder. Note that the encoder method within TreeAutoencoder
    # is exposed as a property which maps to self.model.encode.
    memory = model.encoder(src_tensor)
    
    # Use the model's get_cls_summary to extract the embedding corresponding
    # to the CLS token (which was prepended during encoding).
    cls_embedding = model.get_cls_summary(memory)
    
    return cls_embedding

############################################  
# 9) Main Demo Run Function  
############################################  
def demo_run():  
    if sys.argv[1].lower() == "train":  
        print("Loading dataset...")  
        with open("/scratch/ad7786/chehab-vectorization-rl/dataset_balanced_32_15_5000000.txt", 'r') as f:  
            all_expressions = [x.strip() for x in f.readlines()]  
        # For quick testing, limit the dataset size  
        random.seed(42)
        random.shuffle(all_expressions)
        
        total_samples = len(all_expressions)  
        print(f"Dataset loaded. Total expressions: {total_samples}")  
        train_expr = all_expressions  
        train_dataset = [parse_sexpr(expr) for expr in train_expr]  
        
        if master_process:  
            print(f"Dataset size -> Train: {len(train_dataset)}")  
        model = TreeAutoencoder().to(device)  
        if ddp:  
            model = AEDDP(model, device_ids=[deviceids[ddp_rank]])  
        train_autoenc(model, train_dataset)  
        if ddp:  
            dist.destroy_process_group()  
    elif sys.argv[1].lower() == "test":  
        model = TreeAutoencoder()  
        state_dict = torch.load(sys.argv[2], map_location=device)  
        new_state_dict = {}  
        for key, value in state_dict.items():  
            new_key = key  
            if key.startswith("module."):  
                new_key = key[len("module."): ]  
            new_state_dict[new_key] = value  
        model.load_state_dict(new_state_dict)  
        model.to(device)  
        test_expressions = []  
        with open("llm_collapsed.txt", 'r') as f:  
            for line in f:  
                test_expressions.append(line.strip())  
        test_dataset = [parse_sexpr(expr) for expr in test_expressions][:-1]  
          
        print("Starting evaluation...")  
        final_acc, _ = test_autoenc(model, test_dataset, stdout=True)  
        print(f"\nFinal Test Results:")  
        print(f"Exact Match Accuracy: {final_acc['exact']*100:.2f}%")  
        print(f"Token-level Accuracy: {final_acc['token']*100:.2f}%")  


if __name__ == "__main__":  
    demo_run()
    # model = TreeAutoencoder()  
    # state_dict = torch.load(sys.argv[1], map_location=device)  
    # new_state_dict = {}  
    # for key, value in state_dict.items():  
    #     new_key = key  
    #     if key.startswith("module."):  
    #         new_key = key[len("module."): ]  
    #     new_state_dict[new_key] = value  
    # model.load_state_dict(new_state_dict)  
    # model.to(device)  
    
    
    # exp = parse_sexpr("(Vec (- (* in_0_0 c11_0) (+ in_0_1 c11_1) (- in_0_2 c11_2)) (- (* in_0_3 c11_3) (+ in_0_4 c11_4) (- in_0_5 c11_5)) (+ v10_0 o1) (- v10_1 o2) (* v10_2 o3))")
    # print(get_expression_cls_embedding(exp,model))