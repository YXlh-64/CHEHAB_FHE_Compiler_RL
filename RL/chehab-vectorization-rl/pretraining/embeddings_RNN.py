import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import sys, os
sys.path.append(os.path.abspath("./pytrs"))

from veclang_gen import generate_multiple_expressions, MAX_INT_NUMBER
from pytrs import (Op, VARIABLE_RANGE, CONST_OFFSET,PAREN_CLOSE,PAREN_OPEN, node_to_id, parse_sexpr)

def print(*args, **kwargs):
    kwargs['flush'] = True
    __builtins__.print(*args, **kwargs)

# Set device to GPU 
if torch.cuda.is_available() and sys.argv[1].lower() == "train":
    device = torch.device("cuda")
else :
    device = torch.device("cpu")
print("Using device:", device)


class Config:
    max_gen_length = 1024
    max_depth = 3
    vector_size = 9

    vocab_size = CONST_OFFSET + MAX_INT_NUMBER + 2 + 1 + 1
    start_token = CONST_OFFSET + MAX_INT_NUMBER 
    end_token = CONST_OFFSET + MAX_INT_NUMBER + 1
    pad_token = CONST_OFFSET + MAX_INT_NUMBER + 2
    depth_bits = 2

    # Encoder parameters
    encoder_embed_dim = 256
    encoder_hidden_dim = 256
    encoder_layers = 2
    encoder_dropout = 0.2

    # Decoder parameters
    decoder_embed_dim = 256
    decoder_hidden_dim = 256
    decoder_layers = 2
    decoder_dropout = 0.2

    # Training parameters
    batch_size = 32
    learning_rate = 3e-4
    epochs = 23
    dropout_rate = 0.3

    # Data parameters
    total_samples = int(((1000000 // (0.7*32)) + 1 ) * 32)  # Total dataset size
    train_ratio = 0.7      # 70% training
    valid_ratio = 0.15     # 15% validation
    test_ratio = 0.15      # 15% testing

    # Add to existing parameters
    valid_every = 1        # Validate every N epochs

    @property
    def pos_dim(self):
        return self.depth_bits

config = Config()

def dfs_traverse(expr, depth=0, node_list=None):
    """
    DFS that inserts 'open parentheses' and 'close parentheses' tokens
    around any operator node and its children.
    node_list will be a list of tuples (node_or_paren_symbol, depth).
    """
    if node_list is None:
        node_list = []


    if isinstance(expr, Op):
        # Insert an open parenthesis
        node_list.append((PAREN_OPEN, depth))

        # Insert the operator itself
        node_list.append((expr, depth))

        # Recurse into children
        for child in expr.args:
            dfs_traverse(child, depth + 1, node_list)

        # Insert a close parenthesis
        node_list.append((PAREN_CLOSE, depth))

    else:
        # For Var or Const, just add it directly
        node_list.append((expr, depth))

    return node_list

def bin_encode(val, bits=4):
    """Binary encoding with configurable bit length"""
    cap = 2**bits
    v = min(val, cap-1)
    return [(v >> b) & 1 for b in reversed(range(bits))]

def flatten_expr(expr):
    """
    Flatten expression using a DFS that includes parentheses tokens,
    then map each node or paren symbol to its ID and record its depth.
    """
    node_list = dfs_traverse(expr, 0)   # get (symbol, depth) pairs
    results = []

    varmap = {}
    intmap = {}
    next_var_id = VARIABLE_RANGE[0]
    next_int_id = CONST_OFFSET

    for node_or_paren, depth in node_list:
        # If it's one of the special paren tokens, just use it directly as ID
        if node_or_paren == PAREN_OPEN or node_or_paren == PAREN_CLOSE:
            nid = node_or_paren
        else:
            # Otherwise, it's an Op / Var / Const
            nid, next_var_id, next_int_id, _ = node_to_id(
                node_or_paren, varmap, intmap, next_var_id, next_int_id
            )

        depthvec = bin_encode(depth, config.depth_bits)
        results.append({"node_id": nid, "depth_vec": depthvec})

    return results

class TreeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.node_embed = nn.Embedding(config.vocab_size, config.encoder_embed_dim)
        self.pos_encoder = nn.Sequential(
            nn.Linear(config.pos_dim, config.pos_dim * 2),
            nn.LayerNorm(config.pos_dim * 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate)
        )
        self.rnn = nn.GRU(
            input_size=config.encoder_embed_dim + config.pos_dim * 2,
            hidden_size=config.encoder_hidden_dim,
            num_layers=config.encoder_layers,
            bidirectional=True,
            dropout=config.encoder_dropout if config.encoder_layers > 1 else 0,
            batch_first=True
        )

    def forward(self, seq_nodes, seq_pos):
        node_emb = self.node_embed(seq_nodes)
        pos_emb = self.pos_encoder(seq_pos)
        combined = torch.cat([node_emb, pos_emb], dim=-1)
        _, hidden = self.rnn(combined)
        
        hidden = hidden.view(config.encoder_layers, 2, -1, config.encoder_hidden_dim)[-1]
        return torch.cat([hidden[0], hidden[1]], dim=-1)

class TreeDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.decoder_embed_dim)
        self.rnn = nn.GRU(
            input_size=config.decoder_embed_dim + config.encoder_hidden_dim * 2,
            hidden_size=config.decoder_hidden_dim,
            num_layers=config.decoder_layers,
            dropout=config.decoder_dropout if config.decoder_layers > 1 else 0,
            batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(config.decoder_hidden_dim, config.decoder_hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.decoder_hidden_dim * 2, config.vocab_size)
        )
        self.hidden_proj = nn.Linear(
            config.encoder_hidden_dim * 2,
            config.decoder_layers * config.decoder_hidden_dim
        )

    def forward(self, encoder_out, tgt_seq):
        B, T = tgt_seq.size()
        embedded = self.embed(tgt_seq)
        
        # Project encoder output to initial hidden state
        h0 = self.hidden_proj(encoder_out)
        h0 = h0.view(B, config.decoder_layers, config.decoder_hidden_dim)
        h0 = h0.permute(1, 0, 2).contiguous()
        
        context = encoder_out.unsqueeze(1).repeat(1, T, 1)
        rnn_input = torch.cat([embedded, context], dim=-1)
        
        outputs, _ = self.rnn(rnn_input, h0)
        return self.fc(outputs)

class TreeAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = TreeEncoder()
        self.decoder = TreeDecoder()

    def forward(self, src_nodes, src_pos, tgt_seq):
        encoder_out = self.encoder(src_nodes, src_pos)
        return self.decoder(encoder_out, tgt_seq)

def train_autoenc(model, train_dataset, valid_dataset):
    model.train()
    
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    loss_fn = nn.CrossEntropyLoss(ignore_index=config.pad_token)
    
    best_valid_acc = 0.0
    patience_counter = 0
    
    for epoch in range(config.epochs):
        model.train()
        # Training phase
        random.shuffle(train_dataset)
        total_loss = 0.0
        
        for i in range(0, len(train_dataset), config.batch_size):
            batch = train_dataset[i:i+config.batch_size]
            if not batch:
                continue
                
            # Process batch with padding
            inputs, targets = [], []
            for expr in batch:
                flat = flatten_expr(expr)
                node_ids = [f["node_id"] for f in flat]
                depth_vecs = [f["depth_vec"] for f in flat]
                targets.append([config.start_token] + node_ids + [config.end_token])
                inputs.append((node_ids, depth_vecs))
            
            # Calculate padding lengths
            src_lengths = [len(item[0]) for item in inputs]
            max_src_len = max(src_lengths)
            tgt_lengths = [len(t) for t in targets]
            max_tgt_len = max(tgt_lengths)
            
            # Pad sequences
            padded_nodes = []
            padded_pos = []
            padded_targets = []
            
            for (nodes, pos), target in zip(inputs, targets):
                # Pad source nodes and positions
                node_pad = [config.pad_token] * (max_src_len - len(nodes))
                pos_pad = [[0]*config.pos_dim] * (max_src_len - len(pos))
                padded_nodes.append(nodes + node_pad)
                padded_pos.append(pos + pos_pad)
                
                # Pad targets
                target_pad = [config.pad_token] * (max_tgt_len - len(target))
                padded_targets.append(target + target_pad)
            
            # Convert to tensors and move to device
            src_nodes = torch.tensor(padded_nodes, dtype=torch.long, device=device)
            src_pos = torch.tensor(padded_pos, dtype=torch.float, device=device)
            tgt_seq = torch.tensor(padded_targets, dtype=torch.long, device=device)
            
            optimizer.zero_grad()
            logits = model(src_nodes, src_pos, tgt_seq[:, :-1])
            
            loss = loss_fn(
                logits.view(-1, config.vocab_size),
                tgt_seq[:, 1:].contiguous().view(-1)
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / (len(train_dataset)/config.batch_size)
        
        if (epoch + 1) % config.valid_every == 0:
            valid_results,_ = test_autoenc(model, valid_dataset)
            
            print(f"\nEpoch {epoch+1}/{config.epochs}")
            print(f"Train Loss: {avg_train_loss:.4f}")
            print(f"Validation Exact: {valid_results['exact']*100:.2f}%")
            print(f"Validation Token: {valid_results['token']*100:.2f}%")
            
            # Early stopping check
            if valid_results['exact'] > best_valid_acc:
                best_valid_acc = valid_results['exact']
                patience_counter = 0
                    
                print("Saving the model")
                job_id = os.environ.get('SLURM_JOB_ID', "jobid")
                model_name = f'model_RNN_{job_id}_{config.total_samples}.pth'
                torch.save(model.state_dict(), f"saved_models/{model_name}")
            else:
                patience_counter += 1

            scheduler.step(valid_results['exact'])
        else:
            print(f"\nEpoch {epoch+1}/{config.epochs} | Train Loss: {avg_train_loss:.4f}")

def calculate_accuracy(preds, targets):
    """
    Calculate both exact match accuracy and token-level accuracy.
    Ignores start/end tokens and padding.
    """
    exact_matches = 0
    correct_tokens = 0
    total_tokens = 0
    
    for pred, target in zip(preds, targets):
        # Remove start/end tokens and padding
        clean_pred = [t for t in pred if t not in {config.start_token, config.end_token, config.pad_token}]
        clean_target = [t for t in target if t not in {config.start_token, config.end_token, config.pad_token}]
        
        # Exact match
        exact_matches += int(clean_pred == clean_target)
        
        # Token-level accuracy
        min_len = min(len(clean_pred), len(clean_target))
        if min_len == 0:  # Handle empty sequences
            continue
            
        matches = sum(p == t for p, t in zip(clean_pred[:min_len], clean_target[:min_len]))
        correct_tokens += matches
        total_tokens += len(clean_target)
    
    return {
        'exact': exact_matches / len(preds),
        'token': correct_tokens / total_tokens if total_tokens else 0
    }

def test_autoenc(model, dataset, return_samples=False,stdout=False):
    model.eval()
    all_preds = []
    all_targets = []
    sample_examples = []  # For qualitative logging
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
                depth_vecs = [f["depth_vec"] for f in flat]
                targets.append([config.start_token] + node_ids + [config.end_token])
                inputs.append((node_ids, depth_vecs))
            max_src_len = max(len(item[0]) for item in inputs)
            padded_nodes, padded_pos = [], []
            for nodes, pos in inputs:
                padded_nodes.append(nodes + [config.pad_token]*(max_src_len - len(nodes)))
                padded_pos.append(pos + [[0]*config.pos_dim]*(max_src_len - len(pos)))
            src_nodes = torch.tensor(padded_nodes, dtype=torch.long, device=device)
            src_pos = torch.tensor(padded_pos, dtype=torch.float, device=device)
            encoder_outs = model.encoder(src_nodes, src_pos)
            
            batch_preds = []
            for j in range(len(encoder_outs)):
                cur_input = torch.tensor([[config.start_token]], dtype=torch.long, device=device)
                hidden = model.decoder.hidden_proj(encoder_outs[j].unsqueeze(0))
                hidden = hidden.view(1, config.decoder_layers, config.decoder_hidden_dim)
                hidden = hidden.permute(1, 0, 2).contiguous()
                pred_tokens = []
                for _ in range(config.max_gen_length):
                    embedded = model.decoder.embed(cur_input)
                    context = encoder_outs[j].unsqueeze(0).unsqueeze(0)
                    rnn_input = torch.cat([embedded, context], dim=-1)
                    output, hidden = model.decoder.rnn(rnn_input, hidden)
                    logits = model.decoder.fc(output.squeeze(1))
                    next_token = logits.argmax(-1)
                    token_item = next_token.item()
                    pred_tokens.append(token_item)
                    if token_item == config.end_token:
                        break
                    cur_input = next_token.unsqueeze(0)
                batch_preds.append(pred_tokens)
            all_preds.extend(batch_preds)
            all_targets.extend(targets)
            
            # Print each test expression's target and predicted tokens
            if stdout:
                for orig_expr, target, pred in zip(orig_exprs, targets, batch_preds):
                    print("Expression:", orig_expr)
                    print("Target:    ", target[1:])
                    print("Predicted: ", pred)
                    print("")
                
            # For a few examples, log the raw tokens (as lists) for qualitative review.
            if return_samples and len(sample_examples) < 5:
                for idx in range(min(2, len(batch_preds))):
                    sample_examples.append({
                        "target": targets[idx],
                        "prediction": batch_preds[idx]
                    })
    sample_text = ""
    if sample_examples:
        for ex in sample_examples:
            sample_text += f"Target: {ex['target']}\nPred : {ex['prediction']}\n\n"
    return calculate_accuracy(all_preds, all_targets), sample_text

def demo_run():
    # Generate and split dataset
    
    
    if sys.argv[1].lower() == "train" : 
        all_expressions = generate_multiple_expressions(
            n=config.total_samples,
            max_depth=config.max_depth,
            vector_size=config.vector_size
        )
    
        random.shuffle(all_expressions)
        
        train_end = int(config.total_samples * config.train_ratio)
        valid_end = train_end + int(config.total_samples * config.valid_ratio)
        
        train_expr = all_expressions[:train_end]
        valid_expr = all_expressions[train_end:valid_end]
        test_expr = all_expressions[valid_end:]
        
        train_dataset = [parse_sexpr(expr) for expr in train_expr]
        valid_dataset = [parse_sexpr(expr) for expr in valid_expr]
        test_dataset = [parse_sexpr(expr) for expr in test_expr]
        
        print(f"Dataset sizes: Train={len(train_dataset)}, Valid={len(valid_dataset)}, Test={len(test_dataset)}")
        
        model = TreeAutoencoder().to(device)
        
        train_autoenc(model, train_dataset, valid_dataset)
        
        test_results,_ = test_autoenc(model, test_dataset)
        
        print(f"\nFinal Test Results:")
        print(f"Exact Match Accuracy: {test_results['exact']*100:.2f}%")
        print(f"Token-level Accuracy: {test_results['token']*100:.2f}%")
    elif sys.argv[1].lower() == "test" : 
        model = TreeAutoencoder()
        model.load_state_dict(torch.load(sys.argv[2], map_location=device,weights_only=True))

        # Then move it to the appropriate device (CPU or GPU)
        model.to(device)

        # test = generate_multiple_expressions(
        #     n=100000,
        #     max_depth=config.max_depth,
        #     vector_size=config.vector_size
        # )
        test = []
        with open("test_RNN.txt", 'r') as f:
            for line in f:
                tes = line.strip()
                test.append(tes)
        test = [parse_sexpr(expr) for expr in test]
        
        
        test_results,samples = test_autoenc(model, test,stdout=True)
        
        print(f"\nFinal Test Results:")
        print(f"Exact Match Accuracy: {test_results['exact']*100:.2f}%")
        print(f"Token-level Accuracy: {test_results['token']*100:.2f}%")

if __name__ == "__main__":
    demo_run()