# Neuro Assist Bot

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Aiogram](https://img.shields.io/badge/Aiogram-3.x-brightgreen)

Neuro Assist Bot is an AI-powered Telegram bot built using PyTorch and Aiogram. It uses a neural network to understand and respond to user messages, including opening web pages when certain topics are mentioned.

## Features

- **AI-Powered Responses**: Utilizes a neural network to generate intelligent responses based on user input.
- **Web Integration**: Automatically opens relevant web pages for specific topics like anime.
- **Customizable Intents**: Easily add or modify intents to tailor bot responses.
- **Multilingual Support**: Supports multiple languages thanks to UTF-8 encoding.

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/neuro-assist-bot.git
   cd neuro-assist-bot
2. **Set up the environment**:
   
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
4. **Configure the bot**:
   - Create a .env file with your Telegram bot token:

    ```bash
    BOT_TOKEN=your_telegram_bot_token
5. **Run the bot**:
   ```bash
   python start.py
