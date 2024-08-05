from aiogram import Dispatcher, types, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import Message
dp = Dispatcher()
from aiogram import F
from aiogram.filters import CommandStart, Command
from aiogram.types import Message
import random
from aiogram.types import Message
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import webbrowser, subprocess, os

async def echo(message: Message):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open('intents.json', 'r', encoding='utf-8') as f:
        intents = json.load(f)

    bot_name = "T"
    FILE = "data.pth"

    data = torch.load(FILE, weights_only=True)

    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data["all_words"]
    tags = data["tags"]
    model_state = data["model_state"]

    model = NeuralNet(input_size, hidden_size, output_size).to(device)

    model.load_state_dict(model_state)
    model.eval()
    
    sentence = message.text
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).float().to(device)
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]


    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:

        for intent in intents["intents"]:
            if tag == intent["tag"]:
                if tag == "anime":
                    await message.answer(f'{bot_name}: {random.choice(intent["responses"])}')
                    url = 'https://jut.su/'
                    os.system(f'start {url}')

    else:
        await message.answer(f'{bot_name}: Не понимаю..............')

#-----------------------------------------------------------------------------------------------------------------------------------------
def reg_handler(dp):
    dp.message.register(echo)