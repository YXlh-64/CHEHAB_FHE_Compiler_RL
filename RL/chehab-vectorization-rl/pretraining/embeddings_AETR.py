import torch
import torch.nn as nn
import torch.optim as optim
import math
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
    max_depth = 6
    vector_size = 9

    vocab_size = CONST_OFFSET + MAX_INT_NUMBER + 2 + 1 + 1
    start_token = CONST_OFFSET + MAX_INT_NUMBER 
    end_token = CONST_OFFSET + MAX_INT_NUMBER + 1
    pad_token = CONST_OFFSET + MAX_INT_NUMBER + 2
    depth_bits = math.ceil(math.log2(max_depth + 1))

    # # We won't rely on these old RNN settings, but keep them for continuity
    # encoder_embed_dim = 256
    # encoder_hidden_dim = 256
    # encoder_layers = 2
    # encoder_dropout = 0.2
    # decoder_embed_dim = 256
    # decoder_hidden_dim = 256
    # decoder_layers = 2
    # decoder_dropout = 0.2

    
    d_model = 256
    num_heads = 8
    num_encoder_layers = 4
    num_decoder_layers = 4
    dim_feedforward = 512
    transformer_dropout = 0.2
    max_seq_length = 1500  

   
    batch_size = 32
    learning_rate = 3e-4
    epochs = 25
    dropout_rate = 0.3

    # Data parameters
    total_samples = int(((3000000 // (0.7*32)) + 1 ) * 32)  # Total dataset size
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

#########################################
#      TRANSFORMER-BASED AUTOENCODER    #
#########################################

class PositionalEncoding(nn.Module):
    """
    Learned positional encoding for sequences.
    This is an alternative to the original sinusoidal approach.
    We have an embedding table of shape (max_seq_length, d_model).
    We add the position embedding to token embeddings in forward().
    """
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x, positions=None):
        """
        x: (batch_size, seq_len, d_model)
        positions: (batch_size, seq_len) [optional] containing the positions for each element.
                   If None, assume positions = [0..seq_len-1].
        """
        batch_size, seq_len, d_model = x.shape
        if positions is None:
            positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).repeat(batch_size, 1)
        # shape of pos_emb: (batch_size, seq_len, d_model)
        pos_emb = self.pos_embedding(positions)
        return x + pos_emb


class TransformerAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # We'll create one embedding layer for the tokens (source & target share same vocab).
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_encoder = PositionalEncoding(config.d_model, max_len=config.max_seq_length)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.transformer_dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_encoder_layers)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.transformer_dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.num_decoder_layers)

        # Final projection layer (decoder output -> vocab logits)
        self.output_fc = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, src_nodes, tgt_seq):
        """
        Full forward for training (teacher-forced).
        src_nodes: (batch, src_len), token IDs
        tgt_seq:   (batch, tgt_len), token IDs (including start_token at the beginning)
        """
        # ============== Encoder ==============
        # Embed + positional encode the source
        src_emb = self.token_embedding(src_nodes)  # (batch, src_len, d_model)
        src_emb = self.pos_encoder(src_emb)        # + pos encoding
        # We can optionally create src_key_padding_mask if there's padding
        # but for simplicity, we omit that for now. We'll pass None.
        memory = self.encoder(src_emb)             # (batch, src_len, d_model)

        # ============== Decoder ==============
        # The decoder needs to attend only up to the current position in the target (causal mask).
        tgt_emb = self.token_embedding(tgt_seq)    # (batch, tgt_len, d_model)
        tgt_emb = self.pos_encoder(tgt_emb)
        # Create a causal mask for the target to prevent attention to future tokens.
        tgt_len = tgt_seq.size(1)
        tgt_mask = self.generate_square_subsequent_mask(tgt_len, device=tgt_seq.device)

        decoder_out = self.decoder(
            tgt_emb, memory, 
            tgt_mask=tgt_mask
        )  # (batch, tgt_len, d_model)

        logits = self.output_fc(decoder_out)  # (batch, tgt_len, vocab_size)
        return logits

    @staticmethod
    def generate_square_subsequent_mask(sz, device):
        """
        Generates an upper-triangular matrix of -inf, with zeros on diag.
        This is used as a mask to keep the Transformer decoder from 'seeing' future positions.
        shape: (sz, sz)
        """
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask==0, float('-inf')).masked_fill(mask==1, float(0.0))
        return mask

    def encode(self, src_nodes):
        """Encodes the source sequence into memory for step-by-step decoding (if we want incremental)."""
        src_emb = self.token_embedding(src_nodes)
        src_emb = self.pos_encoder(src_emb)
        memory = self.encoder(src_emb)
        return memory

    def decode_step(self, memory, partial_tgt_seq):
        """
        Decode one step using partial target seq. We'll produce the next-token logits for the final position.
        This is for GREEDY or beam search decoding at inference.
        partial_tgt_seq: (batch, current_length)
        """
        tgt_emb = self.token_embedding(partial_tgt_seq)
        tgt_emb = self.pos_encoder(tgt_emb)
        cur_len = partial_tgt_seq.size(1)
        mask = self.generate_square_subsequent_mask(cur_len, partial_tgt_seq.device)
        dec_out = self.decoder(tgt_emb, memory, tgt_mask=mask)  # (batch, cur_len, d_model)
        # return logits for the last token in the partial sequence
        next_logits = self.output_fc(dec_out[:, -1, :])  # (batch, vocab_size)
        return next_logits


###########################################
#   REPLACE TreeAutoencoder with a new    #
#   Transformer-based autoencoder class   #
###########################################

class TreeAutoencoder(nn.Module):
    """
    We'll keep the same name so your training/test code remains the same,
    but internally it uses a Transformer now.
    """
    def __init__(self):
        super().__init__()
        self.model = TransformerAutoencoder()

    def forward(self, src_nodes, src_pos, tgt_seq):
        # We no longer need 'src_pos' for anything
        # The forward pass now is: (src_nodes, tgt_seq)
        return self.model(src_nodes, tgt_seq)

    @property
    def encoder(self):
        # Expose the encoder for test-time usage
        return self.model.encode

    @property
    def decoder(self):
        # We'll return an object that has decode_step for test-time usage
        return self.model

#######################################################
#   The rest is your existing training/testing code   #
#######################################################

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
                _ = [f["depth_vec"] for f in flat]  # depth info unused in Transformer
                tgt = [config.start_token] + node_ids + [config.end_token]
                inputs.append(node_ids)
                targets.append(tgt)

            src_lengths = [len(x) for x in inputs]
            max_src_len = max(src_lengths)
            tgt_lengths = [len(t) for t in targets]
            max_tgt_len = max(tgt_lengths)

            padded_src = []
            padded_tgt = []
            for src_seq, tgt_seq in zip(inputs, targets):
                src_seq_pad = src_seq + [config.pad_token]*(max_src_len - len(src_seq))
                tgt_seq_pad = tgt_seq + [config.pad_token]*(max_tgt_len - len(tgt_seq))
                padded_src.append(src_seq_pad)
                padded_tgt.append(tgt_seq_pad)
            
            src_nodes = torch.tensor(padded_src, dtype=torch.long, device=device)
            tgt_seq = torch.tensor(padded_tgt, dtype=torch.long, device=device)

            optimizer.zero_grad()
            # Our Transformer-based forward is (src_nodes, tgt_seq[:, :-1]) -> logits
            # We predict the next token for each position
            out_logits = model(src_nodes, None, tgt_seq[:, :-1])  # shape [batch, tgt_len-1, vocab_size]

            # We'll compare out_logits to tgt_seq[:, 1:]
            # Make sure shapes match: out_logits is (batch, seq_len, vocab_size)
            # so shift the target by 1
            loss = loss_fn(
                out_logits.view(-1, config.vocab_size),
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
                model_name = f'model_Transformer_{job_id}_{config.total_samples}.pth'
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
        if len(clean_pred) != len(clean_target):
            print("Length mismatch!")
        matches = sum(p == t for p, t in zip(clean_pred[:min_len], clean_target[:min_len]))
        correct_tokens += matches
        total_tokens += len(clean_target)
    
    return {
        'exact': exact_matches / len(preds),
        'token': correct_tokens / total_tokens if total_tokens else 0
    }

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
            # We'll store ground truth to measure accuracy
            # We do the same approach: remove final token from input to decode
            # But for *greedy decoding*, let's do it step by step

            memory = model.encoder(src_nodes)

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
                    # append next_token to partial_seq
                    partial_seq = torch.cat([partial_seq, torch.tensor([[next_token]], device=device)], dim=1)

                batch_preds.append(pred_tokens)

            all_preds.extend(batch_preds)
            all_targets.extend(padded_tgt)

            # If stdout is True, print some
            if stdout:
                for orig_expr, target, pred in zip(orig_exprs, padded_tgt, batch_preds):
                    print("Expression:", orig_expr)
                    print("Target:    ", target[1:])  # ignoring start_token in display
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

def demo_run():
    if sys.argv[1].lower() == "train":
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

    elif sys.argv[1].lower() == "test":
        model = TreeAutoencoder()
        model.load_state_dict(torch.load(sys.argv[2], map_location=device))
        model.to(device)

        test = []
        with open("test_RNN.txt", 'r') as f:
            for line in f:
                tes = line.strip()
                test.append(tes)
        test = [parse_sexpr(expr) for expr in test]
        
        test_results, samples = test_autoenc(model, test, stdout=True)
        print(f"\nFinal Test Results:")
        print(f"Exact Match Accuracy: {test_results['exact']*100:.2f}%")
        print(f"Token-level Accuracy: {test_results['token']*100:.2f}%")

if __name__ == "__main__":
    demo_run()
