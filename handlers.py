from aiogram import Dispatcher, types
from aiogram.types import Message
import random
import json
import torch
from transformers import BertTokenizer
from model import BERTIntentClassifier  # Импортируйте ваш класс модели
import os, subprocess
# Инициализация устройства
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

# Загрузка intents.json
with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

dp = Dispatcher()
bot_name = "T"

async def echo(message: Message):
    def predict_intent(text):
        with torch.no_grad():
            inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(device)
            outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            _, predicted = torch.max(outputs, dim=1)
            intent = label_encoder.inverse_transform(predicted.cpu().numpy())[0]
        return intent

    # Общение с ботом
    sentence = message.text
    intent_tag = predict_intent(sentence)  # Получаем тег намерения

    if intent_tag == "anime":  # Проверяем, соответствует ли тег нужному
        # Находим соответствующий intent в intents
        for intent in intents['intents']:
            if intent['tag'] == intent_tag:
                intent_responses = intent.get('responses', [])  # Получаем список ответов
                if intent_responses:
                    await message.answer(f'{bot_name}: {random.choice(intent_responses)}')
                    url = 'https://jut.su/'
                    url1 = 'https://aniu.ru/'
                    os.system(f'start {url}')
                    os.system(f'start {url1}')
                break  # Прерываем цикл после обработки нужного тега
    if intent_tag == "launchgenshin":  # Проверяем, соответствует ли тег нужному
        # Находим соответствующий intent в intents
        for intent in intents['intents']:
            if intent['tag'] == intent_tag:
                intent_responses = intent.get('responses', [])  # Получаем список ответов
                if intent_responses:
                    await message.answer(f'{bot_name}: {random.choice(intent_responses)}')
                    await message.answer(f'{bot_name}: Будет открыт лаунчер игры^^')
                    path_to_exe = r"Q:\HoYoPlay\launcher.exe"
                    # Запуск исполняемого файла
                    subprocess.run([path_to_exe])
                break  # Прерываем цикл после обработки нужного тега
    if intent_tag == "closegenshin":  # Проверяем, соответствует ли тег нужному
        # Находим соответствующий intent в intents
        for intent in intents['intents']:
            if intent['tag'] == intent_tag:
                intent_responses = intent.get('responses', [])  # Получаем список ответов
                if intent_responses:
                    await message.answer(f'{bot_name}: {random.choice(intent_responses)}')
                    process_name = 'GenshinImpact.exe'
                    process_name1 = "HYP.exe"
                    # Завершение процесса
                    subprocess.run(['taskkill', '/F', '/IM', process_name])
                    subprocess.run(['taskkill', '/F', '/IM', process_name1])
                break  # Прерываем цикл после обработки нужного тега
    if intent_tag == "launchzzz":  # Проверяем, соответствует ли тег нужному
        # Находим соответствующий intent в intents
        for intent in intents['intents']:
            if intent['tag'] == intent_tag:
                intent_responses = intent.get('responses', [])  # Получаем список ответов
                if intent_responses:
                    await message.answer(f'{bot_name}: {random.choice(intent_responses)}')
                    await message.answer(f'{bot_name}: Будет открыт лаунчер игры^^')
                    path_to_exe = r"Q:\HoYoPlay\launcher.exe"
                    # Запуск исполняемого файла
                    subprocess.run([path_to_exe])
                break  # Прерываем цикл после обработки нужного тега
    if intent_tag == "closezzz":  # Проверяем, соответствует ли тег нужному
        # Находим соответствующий intent в intents
        for intent in intents['intents']:
            if intent['tag'] == intent_tag:
                intent_responses = intent.get('responses', [])  # Получаем список ответов
                if intent_responses:
                    await message.answer(f'{bot_name}: {random.choice(intent_responses)}')
                    process_name = 'ZenlessZoneZero.exe'
                    process_name1 = "HYP.exe"
                    # Завершение процесса
                    subprocess.run(['taskkill', '/F', '/IM', process_name])
                    subprocess.run(['taskkill', '/F', '/IM', process_name1])
                break  # Прерываем цикл после обработки нужного тега
    


    if intent_tag == "spotify":  # Проверяем, соответствует ли тег нужному
        # Находим соответствующий intent в intents
        for intent in intents['intents']:
            if intent['tag'] == intent_tag:
                intent_responses = intent.get('responses', [])  # Получаем список ответов
                if intent_responses:
                    await message.answer(f'{bot_name}: {random.choice(intent_responses)}')
                    path_to_exe = r"C:\Users\fgrls\AppData\Local\Microsoft\WindowsApps\Spotify.exe"
                    # Запуск исполняемого файла
                    subprocess.run([path_to_exe])
                break  # Прерываем цикл после обработки нужного тега
    if intent_tag == "closespotify":  # Проверяем, соответствует ли тег нужному
        # Находим соответствующий intent в intents
        for intent in intents['intents']:
            if intent['tag'] == intent_tag:
                intent_responses = intent.get('responses', [])  # Получаем список ответов
                if intent_responses:
                    await message.answer(f'{bot_name}: {random.choice(intent_responses)}')
                    process_name = 'Spotify.exe'
                    # Завершение процесса
                    subprocess.run(['taskkill', '/F', '/IM', process_name])
                break  # Прерываем цикл после обработки нужного тега




    

         
    else:
        for intent in intents['intents']:
            if intent['tag'] == intent_tag:
                intent_responses = intent.get('responses', [])  # Получаем список ответов
                if intent_responses:
                    await message.answer(f'{bot_name}: {random.choice(intent_responses)}')
                
              # Прерываем цикл после обработки нужного тега
    
      
    
        




#-----------------------------------------------------------------------------------------------------------------------------------------
def reg_handler(dp):
    dp.message.register(echo)