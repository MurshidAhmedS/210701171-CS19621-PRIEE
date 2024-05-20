import numpy as np
import random
import json
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

# Load intents data
with open('intents.json', 'r') as f:
    intents = json.load(f)

# Preprocess data
all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['intent']
    tags.append(tag)
    for pattern in intent['examples']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

X = []
y = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X.append(bag)
    y.append(tags.index(tag))

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define dataset and dataloader
class ChatDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)

train_dataset = ChatDataset(X_train, y_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=5, shuffle=True)

# Define model
input_size = len(X_train[0])
output_size = len(tags)
hidden_size = 50
model = NeuralNet(input_size, hidden_size, output_size)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Early stopping parameters
best_val_loss = float('inf')
patience = 5
counter = 0

# Lists to store loss values for plotting
train_losses = []
val_losses = []
f1_scores = []

# Train the model with early stopping
for epoch in range(20):  # Reduced epochs to 10
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(torch.tensor(X_val, dtype=torch.float32))
        val_loss = criterion(val_outputs, torch.tensor(y_val, dtype=torch.long))
        val_preds = torch.argmax(val_outputs, dim=1)
        f1 = f1_score(y_val, val_preds, average='weighted')
        
    train_losses.append(loss.item())
    val_losses.append(val_loss.item())
    f1_scores.append(f1)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
    else:
        counter += 1

    if counter >= patience:
        print(f'Early stopping at epoch {epoch+1}')
        break

# Create a DataFrame for the table
data = {
    'Epoch': range(1, len(train_losses) + 1),
    'Training Loss': train_losses,
    'Validation Loss': val_losses,
    'F1 Score': f1_scores
}
df = pd.DataFrame(data)

# Print the table
print(df)

# Plotting the training and validation losses
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Losses')
plt.legend()
plt.show()

# Save the compressed model
compressed_model = model.cpu()
torch.save(compressed_model.state_dict(), 'compressed_model.pth')

print('Training complete. Compressed model saved.')
