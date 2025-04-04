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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import time
import matplotlib.pyplot as plt
import wandb


# Set random seed for reproducibility
SEED = 1234
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

wandb.init(project="NLP_202_logistic_regression_imdb", config={
    "epochs": [20, 50],
    "learning_rates": [1e-3, 1e-4],
    "dropout_rates": [0.3, 0.5],
    "batch_sizes": [1, 16, 32, 64, 128],
    "optimizer": "Adam",
    "loss_function": "BCEWithLogitsLoss"
})

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
    wandb.log({"unique_tokens": len(counter)})
    vocab = {word: idx + 2 for idx, (word, _) in enumerate(counter.most_common(max_vocab_size))}
    print(f"Vocabulary size: {len(vocab)}")
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

# Model definition
class LogisticRegression(nn.Module):
    def __init__(self, vocab_size, embed_dim, dropout_rate=0.5):
        super(LogisticRegression, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, x):
        embedded = self.embedding(x)  
        pooled = embedded.mean(dim=1)  
        dropped = self.dropout(pooled)
        return self.fc(dropped).squeeze(1)  
    
# Training function
def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    for texts, labels, _ in tqdm(dataloader):
        texts, labels = texts.to(device), labels.to(device)
        optimizer.zero_grad()
        predictions = model(texts)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

# Evaluation function
def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for texts, labels, _ in dataloader:
            texts, labels = texts.to(device), labels.to(device)
            predictions = model(texts)
            loss = criterion(predictions, labels)
            epoch_loss += loss.item()
            all_preds.extend(torch.round(torch.sigmoid(predictions)).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')

    avg_loss = epoch_loss / len(dataloader)

    wandb.log({
        "valid_loss": avg_loss,
        "valid_accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })

    return epoch_loss / len(dataloader), accuracy, precision, recall, f1

# Load data
train_texts, train_labels = load_imdb_data("./aclImdb/train")
test_texts, test_labels = load_imdb_data("./aclImdb/test")

# Split data into training and validation sets
train_texts, valid_texts = train_texts[:20000], train_texts[20000:]
train_labels, valid_labels = train_labels[:20000], train_labels[20000:]
print(f"Number of training texts: {len(train_texts)}")
print(f"Number of validation texts: {len(valid_texts)}")

# Load or create vocabulary
vocab = load_or_create_vocab(train_texts)
print(f"Vocabulary size: {len(vocab)}")

# Create datasets and dataloaders
train_dataset = IMDBDataset(train_texts, train_labels, vocab)
valid_dataset = IMDBDataset(valid_texts, valid_labels, vocab)
test_dataset = IMDBDataset(test_texts, test_labels, vocab)

batch_sizes = [1, 16, 32, 64, 128]
time_results = []
accuracy_results = []

for batch_size in batch_sizes:
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Model, optimizer, loss function
    model = LogisticRegression(len(vocab), embed_dim=100, dropout_rate=0.5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    # Train model and measure training time
    start_time = time.time()
    for epoch in range(50): 
        train_loss = train_model(model, train_loader, optimizer, criterion, device)
    end_time = time.time()

    # Evaluate on validation set
    valid_loss, valid_acc, _, _, _ = evaluate_model(model, valid_loader, criterion, device)
    
    time_results.append(end_time - start_time)
    accuracy_results.append(valid_acc)

    print(f"Batch size: {batch_size}, Training time: {end_time - start_time:.2f}s, Validation Accuracy: {valid_acc:.4f}")
    wandb.log({
        "batch_size": batch_size,
        "training_time": end_time - start_time,
        "validation_accuracy": valid_acc
    })

# Plot training time vs batch size
plt.figure(figsize=(10, 5))
plt.plot(batch_sizes, time_results, marker='o', color='#B95CF4')
plt.title("Training Time vs Batch Size")
plt.xlabel("Batch Size")
plt.ylabel("Training Time (s)")
plt.grid()
plt.savefig("training_time_vs_batch_size_lr_v2.png")
plt.show()
plt.close()

# Plot validation accuracy vs batch size
plt.figure(figsize=(10, 5))
plt.plot(batch_sizes, accuracy_results, marker='o', color='#FF3659')
plt.title("Validation Accuracy vs Batch Size")
plt.xlabel("Batch Size")
plt.ylabel("Accuracy")
plt.grid()
plt.savefig("validation_accuracy_vs_batch_size_lr_v2.png")
plt.show()
plt.close()

# Load best batch size model and evaluate on test set
best_batch_size = batch_sizes[np.argmax(accuracy_results)]
print(f"Best Batch Size: {best_batch_size}")
wandb.log({"best_batch_size": best_batch_size})

train_loader = DataLoader(train_dataset, batch_size=best_batch_size, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=best_batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=best_batch_size, shuffle=False, collate_fn=collate_fn)

model = LogisticRegression(len(vocab), embed_dim=100, dropout_rate=0.5).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()

for epoch in range(5):
    train_loss = train_model(model, train_loader, optimizer, criterion, device)

test_loss, test_acc, test_precision, test_recall, test_f1 = evaluate_model(model, test_loader, criterion, device)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")
wandb.log({"test_loss": test_loss, "test_accuracy": test_acc, "test_precision": test_precision, "test_recall": test_recall, "test_f1": test_f1})

from itertools import product

# Set hyperparameters for tuning
batch_sizes = [32, 64, 128]
epochs_list = [10, 30]
learning_rates = [1e-3, 1e-4]
dropout_rates = [0.3, 0.5]

best_params = None
best_valid_acc = 0

# Perform hyperparameter tuning
for batch_size, epochs, lr, dropout_rate in product(batch_sizes, epochs_list, learning_rates, dropout_rates):

    print(f"Training with Parameters -> Batch Size: {batch_size}, Epochs: {epochs}, LR: {lr}, Dropout: {dropout_rate}")

    # Create DataLoader with the current batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    wandb.log({
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": lr,
        "dropout_rate": dropout_rate
    })

    # Initialize model
    model = LogisticRegression(len(vocab), embed_dim=100, dropout_rate=dropout_rate).to(device)

    # Use only Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Use only Binary Cross-Entropy loss
    criterion = nn.BCEWithLogitsLoss()

    # Train the model
    for epoch in range(epochs):
        train_loss = train_model(model, train_loader, optimizer, criterion, device)

    # Evaluate on validation set
    valid_loss, valid_acc, _, _, _ = evaluate_model(model, valid_loader, criterion, device)

    print(f"Batch Size: {batch_size}, Epochs: {epochs}, LR: {lr}, Dropout: {dropout_rate} -> Validation Accuracy: {valid_acc:.4f}")

    # Track best hyperparameters
    if valid_acc > best_valid_acc:
        best_valid_acc = valid_acc
        best_params = {
            'batch_size': batch_size,
            'epochs': epochs,
            'learning_rate': lr,
            'dropout_rate': dropout_rate
        }

# Use the best hyperparameters to train the final model
print(f"Best Hyperparameters: {best_params}")
wandb.log(best_params)
final_model = LogisticRegression(len(vocab), embed_dim=100, dropout_rate=best_params['dropout_rate']).to(device)

# Set optimizer and criterion based on best hyperparameters
final_optimizer = torch.optim.Adam(final_model.parameters(), lr=best_params['learning_rate'])
final_criterion = nn.BCEWithLogitsLoss()

# Create DataLoader with the best batch size
final_train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True, collate_fn=collate_fn)
final_valid_loader = DataLoader(valid_dataset, batch_size=best_params['batch_size'], shuffle=False, collate_fn=collate_fn)
final_test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'], shuffle=False, collate_fn=collate_fn)

# Train the final model
for epoch in range(best_params['epochs']):
    final_train_loss = train_model(final_model, final_train_loader, final_optimizer, final_criterion, device)

# Evaluate on test set
test_loss, test_acc, test_precision, test_recall, test_f1 = evaluate_model(final_model, final_test_loader, final_criterion, device)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")
wandb.log({"test_loss": test_loss, "test_accuracy": test_acc, "test_precision": test_precision, "test_recall": test_recall, "test_f1": test_f1})