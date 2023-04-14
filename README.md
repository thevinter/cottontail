# COTTONTAIL

<p align="center">
  <img width="460" height="300" src="cottontail.png">
</p>

This program is a Telegram bot leveraging GPT to provide an AI assistant that can use multiple tools to reply to users in both single and group chat contexts.

## Features:

1. Can assess and respond to user inputs in both single and group chat contexts.
2. Utilizes multiple tools such as Google Search, WolframAlpha, Python REPL, and Bash commands.
3. Supports human intervention for tasks that require human input.
4. Conversations and context-awareness capabilities.

## Dependencies:

The program requires external Python libraries:

To install the necessary dependencies, use the following command:

```bash
pip install -r requirements.txt
```

## Configuration:

1. Rename the .env.example file as .env and fill the necessary API keys 

2. Edit the .config file with your information

### Google API Key

Create the GOOGLE_API_KEY in the [Google Cloud credential console](https://console.cloud.google.com/apis/credentials) and a GOOGLE_CSE_ID using the [Programmable Search Engine](https://programmablesearchengine.google.com/controlpanel/create). Next, it is good to follow the Follow [these](https://stackoverflow.com/questions/37083058/programmatically-searching-google-in-python-using-custom-search) instructions 

## Getting Started

Before you can use the AI Assistant Telegram Bot, you need to create a new bot on Telegram and get your unique API key. Follow these steps:

- Open the Telegram app and search for the "BotFather" bot.
- Start a chat with the BotFather and send the following command: /newbot
- Follow the instructions provided by the BotFather to create your new bot. It will ask you to choose a name and a username for your bot.
- After you've successfully created your bot, the BotFather will provide you with your unique bot API key (also known as the bot token). Save this API key, as you will need it to run your AI Assistant Telegram Bot.

## Usage:

1. Ensure that you have Python 3.7 or later installed.
2. Install the required dependencies.
3. Set up the .env and .config files according to the configuration instructions.
4. Run the bot using the following command:

```bash
python3 main.py
```

5. Interact with the bot on Telegram. The bot can assist you with various tasks by using different tools and can respond in both single and group chat contexts.

## Customization:

You can customize the available tools and functionalities by modifying the configuration file .config according to your preferences.