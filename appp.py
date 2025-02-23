import os
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import pickle

# Load dataset
df = pd.read_excel("Pet Datadet.xlsx")

def extract_features(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    mfcc = torchaudio.transforms.MFCC(sample_rate=sample_rate, n_mfcc=40)(waveform)
    return mfcc.mean(dim=-1).squeeze(0)  # Mean pooling over time

class PetSoundDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.labels = list(set(self.dataframe["Behavior Label"].values))
        self.label_map = {label: i for i, label in enumerate(self.labels)}
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        file_path = self.dataframe.iloc[idx]['File Path']
        label = self.label_map[self.dataframe.iloc[idx]['Behavior Label']]
        features = extract_features(file_path)
        return features, label

def collate_fn(batch):
    features, labels = zip(*batch)
    features = torch.stack(features)
    labels = torch.tensor(labels)
    return features, labels

# Dataset and Dataloader
dataset = PetSoundDataset(df)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

# Model Architecture
class AudioClassifier(nn.Module):
    def __init__(self, input_size=40, hidden_size=128, num_classes=5):
        super(AudioClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        x = self.fc(h_n[-1])
        return x

# Model Training
model = AudioClassifier(num_classes=len(dataset.labels))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    for features, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# Save Model
with open("pet_behavior_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model training complete and saved as pet_behavior_model.pkl")
