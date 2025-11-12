import torch
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import os

def evaluate_model(model, data_loader):
    model.eval()
    all_preds = []
    all_labels = []
    for xb, yb in data_loader:
        with torch.no_grad():
            preds = model(xb).squeeze().cpu().numpy()
        preds = (preds >= 0.5).astype(int)
        all_preds.extend(preds)
        all_labels.extend(yb.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return {'accuracy': accuracy, 'f1': f1}

def plot_acc_f1_vs_seq_len(results, save_dir):
    seq_lens = sorted(results.keys())
    accs = [results[l]['accuracy'] for l in seq_lens]
    f1s = [results[l]['f1'] for l in seq_lens]
    plt.figure()
    plt.plot(seq_lens, accs, marker='o', label='Accuracy')
    plt.plot(seq_lens, f1s, marker='s', label='F1-score (macro)')
    plt.xlabel('Sequence Length')
    plt.ylabel('Score')
    plt.title('Accuracy & F1 vs. Sequence Length')
    plt.legend()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'acc_f1_vs_seq_len.png'))
    plt.close()

def plot_loss_vs_epochs(losses, save_dir, model_name):
    plt.figure()
    plt.plot(range(1, len(losses)+1), losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title(f'Training Loss vs. Epochs ({model_name} model)')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'loss_vs_epochs_{model_name}.png'))
    plt.close()