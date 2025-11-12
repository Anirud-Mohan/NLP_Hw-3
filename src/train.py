import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
from models import SentimentClassifier
from evaluate import evaluate_model, plot_acc_f1_vs_seq_len, plot_loss_vs_epochs
import pandas as pd
import os
import ast

from utils import set_seed
set_seed(42)

# Fixed params
batch_size = 32
loss_fn = nn.BCELoss()
num_epochs = 5

results_dir = '../results/plots'
os.makedirs(results_dir, exist_ok=True)
# Load preprocessed data
def load_data(seq_len):
    train_df = pd.read_csv(f'../data/pre_processed_data/train_data_len{seq_len}.csv')
    test_df = pd.read_csv(f'../data/pre_processed_data/test_data_len{seq_len}.csv')
    
    # Parse sequences from string representation
    train_x = torch.tensor([ast.literal_eval(seq) if isinstance(seq, str) else seq 
                           for seq in train_df['sequence'].values], dtype=torch.long)
    train_y = torch.tensor(train_df['label'].values, dtype=torch.float)
    test_x = torch.tensor([ast.literal_eval(seq) if isinstance(seq, str) else seq 
                          for seq in test_df['sequence'].values], dtype=torch.long)
    test_y = torch.tensor(test_df['label'].values, dtype=torch.float)
    return train_x, train_y, test_x, test_y

experiments = [
# -------------------- RNN Experiments --------------------
    # Vary sequence length (SGD, clipping)
    { "model": "rnn", "activation": "tanh", "seq_len": 25, "optimizer": "SGD", "strategy": "clipping" },
    { "model": "rnn", "activation": "tanh", "seq_len": 50, "optimizer": "SGD", "strategy": "clipping" },
    { "model": "rnn", "activation": "tanh", "seq_len": 100, "optimizer": "SGD", "strategy": "clipping" },
    # Vary sequence length (SGD, no clipping)
    { "model": "rnn", "activation": "tanh", "seq_len": 25, "optimizer": "SGD", "strategy": "no clipping" },
    { "model": "rnn", "activation": "tanh", "seq_len": 50, "optimizer": "SGD", "strategy": "no clipping" },
    { "model": "rnn", "activation": "tanh", "seq_len": 100, "optimizer": "SGD", "strategy": "no clipping" },

    # Vary activations (Adam, clipping)
    { "model": "rnn", "activation": "relu", "seq_len": 25, "optimizer": "Adam", "strategy": "clipping" },
    { "model": "rnn", "activation": "sigmoid", "seq_len": 25, "optimizer": "Adam", "strategy": "clipping" },
    { "model": "rnn", "activation": "tanh", "seq_len": 25, "optimizer": "Adam", "strategy": "clipping" },
    # Vary activations (Adam, no clipping)
    { "model": "rnn", "activation": "relu", "seq_len": 25, "optimizer": "Adam", "strategy": "no clipping" },
    { "model": "rnn", "activation": "sigmoid", "seq_len": 25, "optimizer": "Adam", "strategy": "no clipping" },
    { "model": "rnn", "activation": "tanh", "seq_len": 25, "optimizer": "Adam", "strategy": "no clipping" },

    # Optimizer comparison (RMSprop, clipping)
    { "model": "rnn", "activation": "tanh", "seq_len": 50, "optimizer": "RMSprop", "strategy": "clipping" },
    { "model": "rnn", "activation": "relu", "seq_len": 50, "optimizer": "RMSprop", "strategy": "clipping" },
    { "model": "rnn", "activation": "sigmoid", "seq_len": 50, "optimizer": "RMSprop", "strategy": "clipping" },
    # Optimizer comparison (RMSprop, no clipping)
    { "model": "rnn", "activation": "tanh", "seq_len": 50, "optimizer": "RMSprop", "strategy": "no clipping" },
    { "model": "rnn", "activation": "relu", "seq_len": 50, "optimizer": "RMSprop", "strategy": "no clipping" },
    { "model": "rnn", "activation": "sigmoid", "seq_len": 50, "optimizer": "RMSprop", "strategy": "no clipping" },


    # -------------------- LSTM Experiments --------------------
    # Sequence length tests (Adam)
    { "model": "lstm", "activation": "tanh", "seq_len": 25, "optimizer": "Adam", "strategy": "clipping" },
    { "model": "lstm", "activation": "tanh", "seq_len": 50, "optimizer": "Adam", "strategy": "clipping" },
    { "model": "lstm", "activation": "tanh", "seq_len": 100, "optimizer": "Adam", "strategy": "clipping" },
    { "model": "lstm", "activation": "tanh", "seq_len": 25, "optimizer": "Adam", "strategy": "no clipping" },
    { "model": "lstm", "activation": "tanh", "seq_len": 50, "optimizer": "Adam", "strategy": "no clipping" },
    { "model": "lstm", "activation": "tanh", "seq_len": 100, "optimizer": "Adam", "strategy": "no clipping" },

    # Activation sweep (SGD)
    { "model": "lstm", "activation": "relu", "seq_len": 25, "optimizer": "SGD", "strategy": "clipping" },
    { "model": "lstm", "activation": "sigmoid", "seq_len": 25, "optimizer": "SGD", "strategy": "clipping" },
    { "model": "lstm", "activation": "relu", "seq_len": 25, "optimizer": "SGD", "strategy": "no clipping" },
    { "model": "lstm", "activation": "sigmoid", "seq_len": 25, "optimizer": "SGD", "strategy": "no clipping" },

    # Optimizer comparison (RMSprop)
    { "model": "lstm", "activation": "tanh", "seq_len": 50, "optimizer": "RMSprop", "strategy": "clipping" },
    { "model": "lstm", "activation": "relu", "seq_len": 50, "optimizer": "RMSprop", "strategy": "clipping" },
    { "model": "lstm", "activation": "tanh", "seq_len": 50, "optimizer": "RMSprop", "strategy": "no clipping" },
    { "model": "lstm", "activation": "relu", "seq_len": 50, "optimizer": "RMSprop", "strategy": "no clipping" },


    # -------------------- BiLSTM Experiments --------------------
    # Sequence length tests (Adam)
    { "model": "bilstm", "activation": "tanh", "seq_len": 25, "optimizer": "Adam", "strategy": "clipping" },
    { "model": "bilstm", "activation": "tanh", "seq_len": 50, "optimizer": "Adam", "strategy": "clipping" },
    { "model": "bilstm", "activation": "tanh", "seq_len": 100, "optimizer": "Adam", "strategy": "clipping" },
    { "model": "bilstm", "activation": "tanh", "seq_len": 25, "optimizer": "Adam", "strategy": "no clipping" },
    { "model": "bilstm", "activation": "tanh", "seq_len": 50, "optimizer": "Adam", "strategy": "no clipping" },
    { "model": "bilstm", "activation": "tanh", "seq_len": 100, "optimizer": "Adam", "strategy": "no clipping" },

    # Activation sweep (RMSprop)
    { "model": "bilstm", "activation": "sigmoid", "seq_len": 50, "optimizer": "RMSprop", "strategy": "clipping" },
    { "model": "bilstm", "activation": "sigmoid", "seq_len": 50, "optimizer": "RMSprop", "strategy": "no clipping" },

    # Optimizer comparison (SGD)
    { "model": "bilstm", "activation": "tanh", "seq_len": 25, "optimizer": "SGD", "strategy": "clipping" },
    { "model": "bilstm", "activation": "tanh", "seq_len": 50, "optimizer": "SGD", "strategy": "clipping" },
    { "model": "bilstm", "activation": "tanh", "seq_len": 25, "optimizer": "SGD", "strategy": "no clipping" },
    { "model": "bilstm", "activation": "tanh", "seq_len": 50, "optimizer": "SGD", "strategy": "no clipping" },
]


# For tracking results
all_results = []
model_losses = {}
model_f1s = {}
seq_len_results = {}

for exp in experiments:
    arch = exp["model"]
    act = exp["activation"]
    seq_len = exp["seq_len"]
    opt_name = exp["optimizer"].lower()
    stability = "grad_clip" if exp["strategy"].lower() == "clipping" else None

    print(f"\n=== Running: arch={arch}, act={act}, opt={opt_name}, seq_len={seq_len}, stability={stability} ===")
    
    # Load data for current sequence length
    train_x, train_y, test_x, test_y = load_data(seq_len)
    train_ds = TensorDataset(train_x, train_y)
    test_ds = TensorDataset(test_x, test_y)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    
    # Model params
    vocab_size = train_x.max().item() + 1
    bidirectional = (arch == 'bilstm')
    arch_type = 'lstm' if arch == 'bilstm' else arch
    
    model = SentimentClassifier(
        vocab_size=vocab_size,
        embedding_dim=100,
        hidden_size=64,
        num_layers=2,
        dropout=0.3,
        arch_type=arch_type,
        activation=act,
        bidirectional=bidirectional
    )
    
    # Optimizer
    if opt_name == 'adam':
        optimizer = optim.Adam(model.parameters())
    elif opt_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=0.01)
    elif opt_name == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters())
    else:
        raise ValueError("Unknown optimizer")
    
    # Training loop
    epoch_losses = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        start_time = time.time()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds.squeeze(), yb.float())
            loss.backward()
            if stability == 'grad_clip':
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        train_time = time.time() - start_time
        avg_loss = epoch_loss / len(train_loader.dataset)
        epoch_losses.append(avg_loss)
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            test_metrics = evaluate_model(model, test_loader)
        
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={test_metrics['accuracy']:.4f}, F1={test_metrics['f1']:.4f}, Time={train_time:.2f}s")
    
    # Store results
    key = f"{arch}_{act}_{opt_name}_{seq_len}_{stability}"
    model_losses[key] = epoch_losses
    model_f1s[key] = test_metrics['f1']
    
    all_results.append({
        'architecture': arch,
        'activation': act,
        'optimizer': opt_name,
        'sequence_length': seq_len,
        'stability': stability,
        'final_accuracy': test_metrics['accuracy'],
        'final_f1': test_metrics['f1'],
        'final_loss': epoch_losses[-1],
        'avg_epoch_time': train_time
    })
    
    # Track best result per sequence length
    if seq_len not in seq_len_results:
        seq_len_results[seq_len] = {'accuracy': test_metrics['accuracy'], 'f1': test_metrics['f1']}
    else:
        if test_metrics['f1'] > seq_len_results[seq_len]['f1']:
            seq_len_results[seq_len] = {'accuracy': test_metrics['accuracy'], 'f1': test_metrics['f1']}

# Save all results to CSV
results_df = pd.DataFrame(all_results)
results_df.to_csv('../results/experiment_results.csv', index=False)
print("\n" + "="*60)
print("All results saved to results/experiment_results.csv")
print("="*60)

# Plot accuracy/f1 vs sequence length
plot_acc_f1_vs_seq_len(seq_len_results, results_dir)
print("Saved plot: acc_f1_vs_seq_len.png")

# Plot best and worst models
if model_f1s:
    best_key = max(model_f1s, key=lambda k: model_f1s[k])
    worst_key = min(model_f1s, key=lambda k: model_f1s[k])
    plot_loss_vs_epochs(model_losses[best_key], results_dir, model_name='best')
    plot_loss_vs_epochs(model_losses[worst_key], results_dir, model_name='worst')
    print(f"Saved plot: loss_vs_epochs_best.png")
    print(f"Saved plot: loss_vs_epochs_worst.png")
    print(f"\nBest model: {best_key} (F1={model_f1s[best_key]:.4f})")
    print(f"Worst model: {worst_key} (F1={model_f1s[worst_key]:.4f})")

print("TRAINING COMPLETE!")
