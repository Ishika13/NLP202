import matplotlib
import json
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import spacy
from collections import Counter
import random
import numpy as np
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import wandb
import seaborn as sns

# Set random seed for reproducibility
SEED = 1234
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

wandb.init(project="NLP202_lstm_imdb_v2")

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Download and extract the dataset
if not os.path.exists("./aclImdb"):
    os.system('wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz')
    os.system('tar -xzf aclImdb_v1.tar.gz')

# Load spaCy tokenizer
os.system('python -m spacy download en_core_web_sm')
nlp = spacy.load("en_core_web_sm")

PAD_IDX = 0
UNK_IDX = 1

def tokenize(text):
    return [token.text.lower() for token in nlp(text)]

# Build vocabulary
def build_vocab(texts, max_vocab_size=25_000):
    counter = Counter(token for text in texts for token in tokenize(text))
    print(f"Unique tokens: {len(counter)}")
    vocab = {word: idx + 2 for idx, (word, _) in enumerate(counter.most_common(max_vocab_size))}
    print(f"Vocabulary size: {len(vocab)}")
    wandb.log({"vocab_size": len(vocab)})
    vocab["<pad>"] = PAD_IDX
    vocab["<unk>"] = UNK_IDX
    return vocab

# Load or create vocab.json
def load_or_create_vocab(texts, vocab_file="vocab.json"):
    if os.path.exists(vocab_file):
        with open(vocab_file, "r") as f:
            vocab = json.load(f)
        print(f"Vocabulary loaded from {vocab_file}")
    else:
        vocab = build_vocab(texts)
        with open(vocab_file, "w") as f:
            json.dump(vocab, f, indent=4)
        print(f"Vocabulary created and saved to {vocab_file}")
    return vocab

# Numericalize text
def numericalize(texts, vocab):
    return [[vocab.get(token, UNK_IDX) for token in tokenize(text)] for text in texts]

# Load IMDB dataset
def load_imdb_data(data_dir):
    texts, labels = [], []
    for label_type in ["pos", "neg"]:
        folder = f"{data_dir}/{label_type}"
        for file in os.listdir(folder):
            with open(f"{folder}/{file}", "r", encoding="utf-8") as f:
                texts.append(f.read())
                labels.append(1 if label_type == "pos" else 0)
    print(f"Number of texts: {len(texts)}")
    wandb.log({"num_texts": len(texts)})
    return texts, labels

# Custom Dataset class
class IMDBDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.texts = numericalize(texts, vocab)
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return torch.tensor(self.texts[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float)

# Collate function for DataLoader
def collate_fn(batch):
    texts, labels = zip(*batch)
    lengths = torch.tensor([len(text) for text in texts])
    padded_texts = pad_sequence(texts, batch_first=True, padding_value=PAD_IDX)
    labels = torch.tensor(labels, dtype=torch.float)
    return padded_texts, labels, lengths

# LSTM Model Definition
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout_rate=0.5):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True, dropout=dropout_rate)
        self.fc = nn.Linear(2 * hidden_dim, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hidden, _) = self.lstm(packed_embedded)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        avg_pooled = output.mean(dim=1)  
        dropped = self.dropout(avg_pooled)
        return self.fc(dropped).squeeze(1)

# Train model with logging
def train_model_with_logging(model, train_loader, valid_loader, epochs, lr, wd, batch_size, dropout_rate):
    print(f"Training..")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.BCEWithLogitsLoss()
    
    best_valid_acc = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Wrap the training loop with tqdm
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=True, position=0)
        
        for texts, labels, lengths in train_loader_tqdm:
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(texts, lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Calculate accuracy
            preds = torch.sigmoid(outputs).round()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # Update tqdm with loss
            train_loader_tqdm.set_postfix(loss=loss.item())

        train_loss = total_loss / len(train_loader)
        train_acc = correct / total
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        wandb.log({"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_acc})

        # Validation loop
        model.eval()
        valid_correct = 0
        valid_total = 0
        valid_loss = 0

        with torch.no_grad():
            valid_loader_tqdm = tqdm(valid_loader, desc="Validating", leave=True, position=0)
            for texts, labels, lengths in valid_loader_tqdm:
                texts, labels = texts.to(device), labels.to(device)
                outputs = model(texts, lengths)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
                preds = torch.sigmoid(outputs).round()
                valid_correct += (preds == labels).sum().item()
                valid_total += labels.size(0)

                valid_loader_tqdm.set_postfix(val_loss=loss.item())

        valid_acc = valid_correct / valid_total
        valid_loss /= len(valid_loader)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}")
        wandb.log({"epoch": epoch+1, "train_loss": train_loss, "train_accuracy": train_acc, "valid_loss": valid_loss, "valid_accuracy": valid_acc})
        
        # Save best model
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_model = model.state_dict()

    return best_valid_acc

# Hyperparameter search function
def hyperparameter_search(train_dataset, valid_dataset):
    print("Starting hyperparameter search...")
    batch_sizes = [32, 64, 128, 512]
    learning_rates = [0.001, 0.0001]
    weight_decays = [0.01]
    dropout_rates = [0.3]
    epochs = [20]

    results = []
    
    for batch_size in batch_sizes:
        for lr in learning_rates:
            for wd in weight_decays:
                for dropout_rate in dropout_rates:
                    for epoch in epochs:
                        print(f"Training with batch_size={batch_size}, lr={lr}, weight_decay={wd}, dropout={dropout_rate}, epochs={epoch}")
                        wandb.log({"Training with batch_size": batch_size, "lr": lr, "weight_decay": wd, "dropout": dropout_rate, "epochs": epoch})
                        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
                        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
                        model = LSTMModel(len(vocab), 100, 128, 2, dropout_rate).to(device)
                        start_time = time.time()
                        acc = train_model_with_logging(model, train_loader, valid_loader, epoch, lr, wd, batch_size, dropout_rate)
                        end_time = time.time()
                        wandb.log({"batch_size": batch_size, "lr": lr, "wd": wd, "dropout": dropout_rate, "epochs": epoch, "accuracy": acc, "train_time": end_time - start_time})
                        results.append({"batch_size": batch_size, "lr": lr, "wd": wd, "dropout": dropout_rate, "epochs": epoch, "accuracy": acc, "time": end_time - start_time})

    df = pd.DataFrame(results)
    df.to_csv("hyperparameter_results.csv", index=False)
    wandb.log({"hyperparameter_results": wandb.Table(dataframe=df)})

    # Plot Accuracy vs Batch Size
    plt.figure(figsize=(8, 5))
    for lr in learning_rates:
        subset = df[df["lr"] == lr]
        plt.plot(subset["batch_size"], subset["accuracy"], marker="o", label=f"LR={lr}", color="#ED003E")
    plt.xlabel("Batch Size")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy vs Batch Size")
    plt.savefig("accuracy_vs_batch_size_lstm_v4.png")
    plt.show()

    # Plot Accuracy vs Learning Rate
    plt.figure(figsize=(8, 5))
    for batch_size in batch_sizes:
        subset = df[df["batch_size"] == batch_size]
        plt.plot(subset["lr"], subset["accuracy"], marker="o", label=f"Batch Size={batch_size}", color="#006000")
    plt.xlabel("Learning Rate")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy vs Learning Rate")
    plt.savefig("accuracy_vs_learning_rate_lstm_v5.png")
    plt.show()

    # Plot batch vs train time
    plt.figure(figsize=(8, 5))
    for lr in learning_rates:
        subset = df[df["lr"] == lr]
        plt.plot(subset["batch_size"], subset["time"], marker="o", label=f"LR={lr}", color="#006DFF")
    plt.xlabel("Batch Size")
    plt.ylabel("Training Time (s)")
    plt.legend()
    plt.title("Training Time vs Batch Size")
    plt.savefig("train_time_vs_batch_size_lstm_v5.png")
    plt.show()

    # Best Model Selection
    best_model_params = df.loc[df["accuracy"].idxmax()]
    print("Best Model:", best_model_params)
    wandb.log({"best_model": best_model_params.to_dict()})

# Error Analysis and Detailed Metrics
def enhanced_error_analysis(model, valid_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        valid_loader_tqdm = tqdm(valid_loader, desc="Performing Error Analysis", leave=True)
        for texts, labels, lengths in valid_loader_tqdm:
            texts, labels, lengths = texts.to(device), labels.to(device), lengths.to(device)
            predictions = model(texts, lengths)
            preds = torch.round(torch.sigmoid(predictions)).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    # Generate classification report
    report = classification_report(all_labels, all_preds, target_names=["Negative", "Positive"])
    print("Classification Report:\n", report)
    wandb.log({"classification_report": report})

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print(f"Confusion Matrix:\n{cm}")
    wandb.log({"confusion_matrix": cm})

    # Visualizing confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Pastel2", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix_lstm_v5.png")
    plt.show()

    # False Positives and False Negatives
    false_positives = np.where((all_preds == 1) & (all_labels == 0))[0]
    false_negatives = np.where((all_preds == 0) & (all_labels == 1))[0]
    print(f"False Positives: {len(false_positives)}")
    print(f"False Negatives: {len(false_negatives)}")
    wandb.log({"false_positives": len(false_positives), "false_negatives": len(false_negatives)})

# Load data
train_texts, train_labels = load_imdb_data("./aclImdb/train")
valid_texts, valid_labels = load_imdb_data("./aclImdb/test")

# Create vocab and datasets
vocab = load_or_create_vocab(train_texts)
train_dataset = IMDBDataset(train_texts, train_labels, vocab)
valid_dataset = IMDBDataset(valid_texts, valid_labels, vocab)

# Run hyperparameter search
best_params = hyperparameter_search(train_dataset, valid_dataset)
wandb.log({"best_params": best_params})

# Recreate the best model
best_model = LSTMModel(
    vocab_size=len(vocab),
    embed_dim=100,
    hidden_dim=128,
    num_layers=2,
    dropout_rate=best_params["dropout"]
).to(device)

# Load validation data loader with the best batch size
best_batch_size = int(best_params["batch_size"])
valid_loader = DataLoader(valid_dataset, batch_size=best_batch_size, shuffle=False, collate_fn=collate_fn)

def evaluate_test_set(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for texts, labels, lengths in test_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts, lengths)
            preds = torch.sigmoid(outputs).round()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    test_accuracy = correct / total
    print(f"Final Test Accuracy: {test_accuracy}")
    wandb.log({"final_test_accuracy": test_accuracy})

# Load the test dataset
test_texts, test_labels = load_imdb_data("./aclImdb/test")
test_dataset = IMDBDataset(test_texts, test_labels, vocab)
test_loader = DataLoader(test_dataset, batch_size=best_batch_size, shuffle=False, collate_fn=collate_fn)

# Evaluate on the test set
evaluate_test_set(best_model, test_loader)

# Perform error analysis
enhanced_error_analysis(best_model, valid_loader)
wandb.log({"enhanced_error_analysis": "completed"})
wandb.finish()