import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer
from model import BERTIntentClassifier  # Импортируем модель
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Загрузка данных
with open('intents.json', 'r') as f:
    intents = json.load(f)

sentences = []
tags = []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        sentences.append(pattern)
        tags.append(intent['tag'])

# Инициализация BERT токенайзера и модели
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Закодируем теги (intents)
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(tags)
num_labels = len(label_encoder.classes_)

model = BERTIntentClassifier(num_labels)  # Создаем модель с количеством меток
model.to(device)

# Пример кодирования с использованием BERT токенайзера
encoded_inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

input_ids = encoded_inputs['input_ids']
attention_mask = encoded_inputs['attention_mask']
labels = torch.tensor(labels)

dataset = TensorDataset(input_ids, attention_mask, labels)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

optimizer = optim.Adam(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()
epochs = 1000
model.train()

for epoch in range(epochs):
    total_loss = 0
    for batch in train_loader:
        input_ids, attention_mask, labels = [item.to(device) for item in batch]
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f'Epochs {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')

# Сохранение модели и меток
torch.save({
    'model_state_dict': model.state_dict(),
    'label_encoder': label_encoder
}, 'bert_intent_classifier.pth')