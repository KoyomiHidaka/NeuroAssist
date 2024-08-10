import torch
import torch.nn as nn
import json
import random
from transformers import BertTokenizer
from model import BERTIntentClassifier  # Импортируйте ваш класс модели

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Инициализация токенизатора
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Создание экземпляра модели с нужным количеством меток
num_labels = 12  # Установите это значение в соответствии с вашими метками
model = BERTIntentClassifier(num_labels=num_labels).to(device)

# Загрузка обученной модели и меток
try:
    checkpoint = torch.load('bert_intent_classifier.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    label_encoder = checkpoint['label_encoder']
except FileNotFoundError:
    print("Checkpoint file not found.")
    exit()
except KeyError as e:
    print(f"Key error: {e}")
    exit()

model.eval()

with open('intents.json', 'r') as f:
    intents = json.load(f)

def predict_intent(text):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(device)
        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        _, predicted = torch.max(outputs, dim=1)
        intent = label_encoder.inverse_transform(predicted.cpu().numpy())[0]
    return intent

# Общение с ботом
bot_name = "T"
print("Go Chat type 'Quit' for exit")

while True:
    sentence = input("You: ")
    if sentence.lower() == "quit":
        break
    
    intent = predict_intent(sentence)
    
    for i in intents['intents']:
        if i['tag'] == intent:
            print(f"{bot_name}: {random.choice(i['responses'])}")
            break