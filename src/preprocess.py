import numpy as np
import pandas as pd
import re 
import os
from collections import Counter

import nltk
import kagglehub
import shutil
nltk.download('punkt')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

from utils import set_seed
set_seed(42)

def load_imdb_data(raw_data_dir):
    os.makedirs(raw_data_dir, exist_ok=True)
    csv_path = os.path.join(raw_data_dir, "IMDB_Dataset.csv")
    
    if not os.path.exists(csv_path):
        print("Downloading IMDB dataset...")
        path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
        print(f"Dataset downloaded to: {path}")

        downloaded_csv = os.path.join(path, "IMDB Dataset.csv")
        shutil.copy(downloaded_csv, csv_path)
        print(f"Dataset copied to: {csv_path}")
    else:
        print("IMDB dataset already exists.")
    
    return csv_path

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text


def tokenize(text):
    return nltk.word_tokenize(text)
    

def build_vocabulary(texts, vocab_size=10000):
    counter = Counter()
    for text in texts:
        tokens = tokenize(text)
        counter.update(tokens)
    top_frequent = counter.most_common(vocab_size)
    word_to_idx = {word: idx +2 for idx, (word, _) in enumerate(top_frequent)}
    word_to_idx["<PAD>"] = 0
    word_to_idx["<UNK>"] = 1
    return word_to_idx

def texts_to_sequences(texts, word_to_idx):
    sequences = []
    for text in texts:
        tokens = tokenize(text)
        seq = [word_to_idx.get(token, word_to_idx["<UNK>"]) for token in tokens]
        sequences.append(seq)
    return sequences


def pad_sequences(sequences, max_length):
    padded_sequences = []
    for seq in sequences:
        if len(seq) < max_length:
            padded_seq = seq + [0] * (max_length - len(seq))
        else:
            padded_seq = seq[:max_length]
        padded_sequences.append(padded_seq)
    return padded_sequences


def preprocess_data(raw_data_dir, output_dir, vocab_size=10000, max_length=100):
    # 1. Read raw CSV
    csv_path = os.path.join(raw_data_dir, "IMDB_Dataset.csv")
    df = pd.read_csv(csv_path)
    reviews = df["review"].tolist()
    sentiments = df["sentiment"].tolist()
    labels = [1 if s == "positive" else 0 for s in sentiments]

    # 2. Split into train/test (first 25k train, last 25k test)
    train_texts = reviews[:25000]
    train_labels = labels[:25000]
    test_texts = reviews[25000:]
    test_labels = labels[25000:]

    # 3. Clean and tokenize
    train_clean = [clean_text(text) for text in train_texts]
    test_clean = [clean_text(text) for text in test_texts]
    train_tokens = [' '.join(tokenize(text)) for text in train_clean]

    # 4. Build vocabulary
    word_to_idx = build_vocabulary(train_tokens, vocab_size)

    # 5. Dataset statistics for reporting 
    avg_train_len = np.mean([len(tokenize(text)) for text in train_clean])
    avg_test_len = np.mean([len(tokenize(text)) for text in test_clean])
    vocab_size_stat = len(word_to_idx)
    print(f"Average train review length: {avg_train_len:.2f}")
    print(f"Average test review length: {avg_test_len:.2f}")
    print(f"Vocabulary size: {vocab_size_stat}")

    # 6. Convert to sequences
    train_seqs = texts_to_sequences(train_clean, word_to_idx)
    test_seqs = texts_to_sequences(test_clean, word_to_idx)

    # 7. Pad sequences
    train_seqs = pad_sequences(train_seqs, max_length)
    test_seqs = pad_sequences(test_seqs, max_length)

    # 8. Save processed data
    save_preprocessed_data((train_seqs, train_labels), os.path.join(output_dir, "train_data.csv"))
    save_preprocessed_data((test_seqs, test_labels), os.path.join(output_dir, "test_data.csv"))

    return train_seqs, train_labels, test_seqs, test_labels, word_to_idx
    


def save_preprocessed_data(data, filepath):
    """Save preprocessed data to disk."""
    sequences, labels = data
    df = pd.DataFrame({'sequence': sequences, 'label': labels})
    df.to_csv(filepath, index=False)


def load_preprocessed_data(filepath):
    """Load preprocessed data from disk."""
    df = pd.read_csv(filepath)
    sequences = df['sequence'].tolist()
    labels = df['label'].tolist()
    return sequences, labels


if __name__ == "__main__":
    raw_data_dir = os.path.join(PROJECT_ROOT, "data", "raw_data")
    output_dir = os.path.join(PROJECT_ROOT, "data", "pre_processed_data")
    load_imdb_data(raw_data_dir)
    # Process for all 3 sequence lengths
    for seq_length in [25, 50, 100]:
        print('='*50)
        print(f"Processing with sequence length: {seq_length}")
        print('='*50)
        
        train_seqs, train_labels, test_seqs, test_labels, word_to_idx = preprocess_data(
            raw_data_dir, output_dir, vocab_size=10000, max_length=seq_length
        )
        
        # Save with length-specific filenames
        save_preprocessed_data(
            (train_seqs, train_labels), 
            os.path.join(output_dir, f"train_data_len{seq_length}.csv")
        )
        save_preprocessed_data(
            (test_seqs, test_labels), 
            os.path.join(output_dir, f"test_data_len{seq_length}.csv")
        )
    
    print("\nPreprocessing complete for all sequence lengths!")