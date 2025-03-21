import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import re
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


# Preprocessing functions
def preproc(x):
    x = re.sub(r"[^a-zA-Z0-9\s]", "", x)
    return x.lower()


def preproc_spanish(text):
    text = re.sub(r"[^a-zA-Z0-9áéíóúüñÁÉÍÓÚÜÑ\s]", "", text)
    return text.lower()


# Load dataset
df = pd.read_csv("spa.txt.csv")
df["English"] = df["English"].apply(preproc)
df["Translated"] = df["Translated"].apply(preproc_spanish)

eng_sentences = df["English"].values
spa_sentences = df["Translated"].values

# Tokenization
eng_tokenizer = Tokenizer(filters="")
spa_tokenizer = Tokenizer(filters="")
eng_tokenizer.fit_on_texts(eng_sentences)
spa_tokenizer.fit_on_texts(spa_sentences)

eng_vocab_size = len(eng_tokenizer.word_index) + 1
spa_vocab_size = len(spa_tokenizer.word_index) + 1

eng_sequences = eng_tokenizer.texts_to_sequences(eng_sentences)
spa_sequences = spa_tokenizer.texts_to_sequences(spa_sentences)

max_eng_len = max(len(seq) for seq in eng_sequences)
max_spa_len = max(len(seq) for seq in spa_sequences)

eng_padded = pad_sequences(eng_sequences, maxlen=max_eng_len, padding="post")
spa_padded = pad_sequences(spa_sequences, maxlen=max_spa_len, padding="post")

# Convert to Torch tensors
eng_tensor = torch.tensor(eng_padded, dtype=torch.long)
spa_tensor = torch.tensor(spa_padded, dtype=torch.long)

dataset = TensorDataset(eng_tensor, spa_tensor[:, :-1], spa_tensor[:, 1:])

# Split dataset into training and validation sets (90% train, 10% validation)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size]
)

train_loader = DataLoader(train_dataset, batch_size=192, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=192, shuffle=False)


# Define Encoder
class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        x = self.embedding(x)
        output, (h, c) = self.lstm(x)
        return output, h, c


# Define Bahdanau Attention
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(BahdanauAttention, self).__init__()
        self.W1 = nn.Linear(hidden_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)

    def forward(self, query, values):
        query = query.squeeze(0).unsqueeze(1)
        score = self.V(torch.tanh(self.W1(query) + self.W2(values)))
        attention_weights = torch.softmax(score, dim=1)
        context_vector = torch.sum(attention_weights * values, dim=1)
        return context_vector


# Define Decoder
class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim + hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.attention = BahdanauAttention(hidden_dim)

    def forward(self, x, hidden, enc_output):
        context_vector = self.attention(hidden[0], enc_output)
        x = self.embedding(x)
        x = torch.cat([context_vector.unsqueeze(1), x], dim=-1)
        output, (h, c) = self.lstm(x, hidden)
        return self.fc(output), (h, c)


# Define Seq2Seq Model
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, inputs, targets):
        enc_output, h, c = self.encoder(inputs)
        dec_hidden = (h, c)
        dec_input = targets[:, 0].unsqueeze(1)
        all_predictions = []

        for t in range(targets.shape[1]):
            predictions, dec_hidden = self.decoder(dec_input, dec_hidden, enc_output)
            all_predictions.append(predictions)
            dec_input = targets[:, t].unsqueeze(1)

        return torch.cat(all_predictions, dim=1)


# Initialize model
embedding_dim = 256
hidden_dim = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Move model to GPU
encoder = Encoder(eng_vocab_size, embedding_dim, hidden_dim).to(device)
decoder = Decoder(spa_vocab_size, embedding_dim, hidden_dim).to(device)
model = Seq2Seq(encoder, decoder).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters())


# Validation function
def validate(model, val_loader, criterion):
    """Evaluates the model on the validation set."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets_in, targets_out in val_loader:
            inputs, targets_in, targets_out = (
                inputs.to(device),
                targets_in.to(device),
                targets_out.to(device),
            )

            outputs = model(inputs, targets_in)
            loss = criterion(outputs.view(-1, spa_vocab_size), targets_out.view(-1))
            total_loss += loss.item()

            # Compute accuracy
            predictions = outputs.argmax(dim=-1)
            mask = targets_out != 0
            correct += (predictions == targets_out).masked_select(mask).sum().item()
            total += mask.sum().item()

    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total if total > 0 else 0
    print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

    return avg_loss, accuracy


# Training function
def train(model, train_loader, val_loader, criterion, optimizer, epochs=50):
    """Trains the model with validation."""
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for inputs, targets_in, targets_out in progress_bar:
            inputs, targets_in, targets_out = (
                inputs.to(device),
                targets_in.to(device),
                targets_out.to(device),
            )

            optimizer.zero_grad()
            outputs = model(inputs, targets_in)

            loss = criterion(outputs.view(-1, spa_vocab_size), targets_out.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Compute accuracy
            predictions = outputs.argmax(dim=-1)
            mask = targets_out != 0
            correct += (predictions == targets_out).masked_select(mask).sum().item()
            total += mask.sum().item()

            progress_bar.set_postfix(
                loss=loss.item(), accuracy=correct / total if total > 0 else 0
            )

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

        # Run validation step
        validate(model, val_loader, criterion)


# Train model
train(model, train_loader, val_loader, criterion, optimizer)

# Save model
torch.save(model.state_dict(), "seq2seq_model.pth")
