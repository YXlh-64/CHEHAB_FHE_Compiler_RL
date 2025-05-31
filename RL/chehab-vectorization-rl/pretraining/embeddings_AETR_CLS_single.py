import torch
import torch.nn as nn
import torch.optim as optim
import math
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

############################################
#  1) We define config.cls_token here and  #
#     increment the vocab by +1            #
############################################
class Config:
    max_gen_length = 1024
    max_depth = 32
    vector_size = 9

    vocab_size = CONST_OFFSET + MAX_INT_NUMBER + 2 + 1 + 1
    start_token = CONST_OFFSET + MAX_INT_NUMBER 
    end_token = CONST_OFFSET + MAX_INT_NUMBER + 1
    pad_token = CONST_OFFSET + MAX_INT_NUMBER + 2
    depth_bits = math.ceil(math.log2(max_depth + 1))


    # Add a new CLS token
    cls_token = vocab_size
    vocab_size += 1  # increment the vocab size by 1 to include CLS

    d_model = 256
    num_heads = 8
    num_encoder_layers = 4
    num_decoder_layers = 4
    dim_feedforward = 512 # changed from 512 to 128
    transformer_dropout = 0.2
    max_seq_length = 1500  

    batch_size = 32
    learning_rate = 3e-4
    epochs = 25
    dropout_rate = 0.3

    # Data parameters
    total_samples = int(((1000000 // (0.7*32)) + 1 ) * 32)  # Total dataset size
    train_ratio = 0.7      # 70% training
    valid_ratio = 0.15     # 15% validation
    test_ratio = 0.15      # 15% testing

    valid_every = 1        # Validate every N epochs

    @property
    def pos_dim(self):
        return self.depth_bits

config = Config()

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

def bin_encode(val, bits=4):
    cap = 2**bits
    v = min(val, cap-1)
    return [(v >> b) & 1 for b in reversed(range(bits))]

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
        depthvec = bin_encode(depth, config.depth_bits)
        results.append({"node_id": nid, "depth_vec": depthvec})

    return results

###################################################
#  2) PositionalEncoding (unchanged from before)  #
###################################################
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

###############################################
#  3) TransformerAutoencoder with [CLS] in   #
#     the encoder input                      #
###############################################
class TransformerAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Extended vocab includes the CLS token
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_encoder = PositionalEncoding(config.d_model, max_len=config.max_seq_length)

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.transformer_dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_encoder_layers)

        # Decoder
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
        mask = mask.float().masked_fill(mask==0, float('-inf')).masked_fill(mask==1, float(0.0))
        return mask

    def forward(self, src_nodes, tgt_seq):
        """
        Standard training forward with teacher forcing:
        - We insert CLS token at start of src_nodes in the encoder
        - Then do usual decoder pass
        """
        # Insert [CLS] at the front of src_nodes
        # src_nodes: (batch, src_len) => new shape: (batch, src_len+1)
        batch_size = src_nodes.size(0)
        cls_column = torch.full((batch_size, 1), config.cls_token, dtype=torch.long, device=src_nodes.device)
        src_nodes_with_cls = torch.cat([cls_column, src_nodes], dim=1)

        # ----- Encoder -----
        src_emb = self.token_embedding(src_nodes_with_cls)
        src_emb = self.pos_encoder(src_emb)
        memory = self.encoder(src_emb)  # shape: (batch, src_len+1, d_model)

        # ----- Decoder -----
        tgt_emb = self.token_embedding(tgt_seq)
        tgt_emb = self.pos_encoder(tgt_emb)
        tgt_len = tgt_seq.size(1)
        tgt_mask = self.generate_square_subsequent_mask(tgt_len, device=tgt_seq.device)

        decoder_out = self.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask
        )
        logits = self.output_fc(decoder_out)  # (batch, tgt_len, vocab_size)
        return logits

    def encode(self, src_nodes):
        """
        For inference or downstream usage: produce memory + also store the CLS vector.
        We'll also insert CLS token at front here.
        """
        batch_size = src_nodes.size(0)
        cls_column = torch.full((batch_size, 1), config.cls_token, dtype=torch.long, device=src_nodes.device)
        src_nodes_with_cls = torch.cat([cls_column, src_nodes], dim=1)

        src_emb = self.token_embedding(src_nodes_with_cls)
        src_emb = self.pos_encoder(src_emb)
        memory = self.encoder(src_emb)
        return memory  # shape: (batch, seq_len+1, d_model)

    def decode_step(self, memory, partial_tgt_seq):
        """
        For token-by-token decoding. The memory already has [CLS] at index 0.
        partial_tgt_seq is shape (batch, cur_len).
        We'll generate the next token logits for the final position.
        """
        tgt_emb = self.token_embedding(partial_tgt_seq)
        tgt_emb = self.pos_encoder(tgt_emb)
        cur_len = partial_tgt_seq.size(1)
        mask = self.generate_square_subsequent_mask(cur_len, partial_tgt_seq.device)
        dec_out = self.decoder(tgt_emb, memory, tgt_mask=mask)
        next_logits = self.output_fc(dec_out[:, -1, :])  # (batch, vocab_size)
        return next_logits

    def get_cls_vector(self, memory):
        """
        If you want to extract just the CLS token's hidden state (the 'summary'),
        memory is shape: (batch, seq_len+1, d_model).
        The CLS hidden state is memory[:, 0, :].
        """
        return memory[:, 0, :]  # shape: (batch, d_model)


#####################################################
#  4) A wrapper that the rest of your code expects  #
#####################################################
class TreeAutoencoder(nn.Module):
    """
    The rest of your training and testing code calls:
      - model(src_nodes, src_pos, tgt_seq)
      - model.encoder(...)  # property
      - model.decoder.decode_step(...)
    We'll integrate the [CLS] approach inside.
    """
    def __init__(self):
        super().__init__()
        self.model = TransformerAutoencoder()

    def forward(self, src_nodes, src_pos, tgt_seq):
        # src_pos is unused in this approach
        return self.model(src_nodes, tgt_seq)

    @property
    def encoder(self):
        return self.model.encode

    @property
    def decoder(self):
        return self.model

    def get_cls_summary(self, memory):
        """
        Optional helper if you want to retrieve the
        (batch_size, d_model) CLS embedding from the memory.
        """
        return self.model.get_cls_vector(memory)

###################################
#   Rest of your training code    #
###################################

def train_autoenc(model, train_dataset, valid_dataset):
    model.train()
    
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    loss_fn = nn.CrossEntropyLoss(ignore_index=config.pad_token)
    
    best_valid_acc = 0.0
    patience_counter = 0
    
    for epoch in range(config.epochs):
        model.train()
        random.shuffle(train_dataset)
        total_loss = 0.0
        
        for i in range(0, len(train_dataset), config.batch_size):
            batch = train_dataset[i:i+config.batch_size]
            if not batch:
                continue

            inputs, targets = [], []
            for expr in batch:
                flat = flatten_expr(expr)
                node_ids = [f["node_id"] for f in flat]
                # We still do (start_token) + nodes + (end_token) for the target
                tgt = [config.start_token] + node_ids + [config.end_token]
                inputs.append(node_ids)
                targets.append(tgt)

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
            tgt_seq = torch.tensor(padded_tgt, dtype=torch.long, device=device)

            optimizer.zero_grad()
            # We feed (src_nodes, tgt_seq[:, :-1]) => predict next token
            logits = model(src_nodes, None, tgt_seq[:, :-1])  # shape [batch, (tgt_len-1), vocab_size]

            loss = loss_fn(
                logits.reshape(-1, config.vocab_size),
                tgt_seq[:, 1:].contiguous().view(-1)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()

        avg_train_loss = total_loss / (len(train_dataset)/config.batch_size)
        
        if (epoch + 1) % config.valid_every == 0:
            valid_results, _ = test_autoenc(model, valid_dataset)
            print(f"\nEpoch {epoch+1}/{config.epochs}")
            print(f"Train Loss: {avg_train_loss:.4f}")
            print(f"Validation Exact: {valid_results['exact']*100:.2f}%")
            print(f"Validation Token: {valid_results['token']*100:.2f}%")
            
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
    exact_matches = 0
    correct_tokens = 0
    total_tokens = 0
    
    for pred, target in zip(preds, targets):
        clean_pred = [t for t in pred if t not in {config.start_token, config.end_token, config.pad_token, config.cls_token}]
        clean_target = [t for t in target if t not in {config.start_token, config.end_token, config.pad_token, config.cls_token}]
        
        exact_matches += int(clean_pred == clean_target)
        
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
                    if next_token == config.end_token or next_token == config.pad_token:
                        pred_tokens.append(config.end_token)
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
        
        test_results, _ = test_autoenc(model, test_dataset)
        print(f"\nFinal Test Results:")
        print(f"Exact Match Accuracy: {test_results['exact']*100:.2f}%")
        print(f"Token-level Accuracy: {test_results['token']*100:.2f}%")

    elif sys.argv[1].lower() == "test":
        model = TreeAutoencoder()
        model.load_state_dict(torch.load(sys.argv[2], map_location=device))
        model.to(device)

        test = []
        with open("test_data.txt", 'r') as f:
            for line in f:
                tes = line.strip()
                test.append(tes)
        test = [parse_sexpr(expr) for expr in test]
        
        test_results, samples = test_autoenc(model, test, stdout=True)
        print(f"\nFinal Test Results:")
        print(f"Exact Match Accuracy: {test_results['exact']*100:.2f}%")
        print(f"Token-level Accuracy: {test_results['token']*100:.2f}%")
        
        
def get_expression_cls_embedding(expr, model):
    """
    Given an expression and a trained TreeAutoencoder model,
    this function returns the CLS embedding of the expression.

    Parameters:
      expr: The expression to process (e.g. a parsed s-expression)
      model: An instance of TreeAutoencoder (or any model with .encoder and .get_cls_summary)

    Returns:
      A tensor of shape (1, d_model) corresponding to the CLS embedding.
    """
    # Flatten the expression into a list of tokens (node ids)
    flat = flatten_expr(expr)
    node_ids = [entry["node_id"] for entry in flat]
    
    # Convert the list of node ids into a tensor with batch size 1
    src_nodes = torch.tensor([node_ids], dtype=torch.long, device=device)
    
    # Use the model's encoder which automatically prepends the CLS token
    memory = model.encoder(src_nodes)  # shape: (1, seq_len+1, d_model)
    
    # Extract and return the CLS token's hidden state (first token)
    cls_embedding = model.get_cls_summary(memory)  # shape: (1, d_model)
    return cls_embedding
if __name__ == "__main__":
    # model = TreeAutoencoder()
    # model.load_state_dict(torch.load("saved_models/model_Transformer_10222304_1428576.pth", map_location=device))
    # model.to(device)
    
    
    # exp = parse_sexpr("(Vec (- (* in_0_0 c11_0) (+ in_0_1 c11_1) (- in_0_2 c11_2)) (- (* in_0_3 c11_3) (+ in_0_4 c11_4) (- in_0_5 c11_5)) (+ v10_0 o1) (- v10_1 o2) (* v10_2 o3))")
    # print(get_expression_cls_embedding(exp,model))
    
    demo_run()