import configparser
import logging
import os
import asyncio
from queue import Queue
from typing import Any, Dict, List, Optional, Union

import openai
from dotenv import load_dotenv
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.callbacks.base import BaseCallbackHandler, CallbackManager
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from langchain.schema import AgentAction, AgentFinish, LLMResult, SystemMessage
from langchain.tools.human.tool import HumanInputRun
from langchain.utilities import BashProcess, GoogleSearchAPIWrapper, PythonREPL
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)

logger = logging.getLogger(__name__)

load_dotenv()
config = configparser.ConfigParser()
config.read('.config')

openai.api_key = os.getenv("OPENAI_API_KEY")

TOKEN = os.getenv("TELEGRAM_TOKEN")
IS_GROUP = config.getboolean("system", "group")
BOT_NAME = config.get("system", "botname")
USERNAME = config.get("user", "username")
LANGUAGE = config.get("system", "language")
IS_AWAITING = False
ALLOWED_CHATS = filters.Chat(chat_id=["253580370"])

ENABLE_HUMAN = config.getboolean("tools", "enable_human")
ENABLE_GOOGLE = config.getboolean("tools", "enable_google")
ENABLE_WOLFRAM = config.getboolean("tools", "enable_wolfram")
ENABLE_BASH = config.getboolean("tools", "enable_bash")
ENABLE_PYTHON = config.getboolean("tools", "enable_python")

search = GoogleSearchAPIWrapper() if ENABLE_GOOGLE else None
human = HumanInputRun() if ENABLE_HUMAN else None
wolfram = WolframAlphaAPIWrapper() if ENABLE_WOLFRAM else None
python = PythonREPL() if ENABLE_PYTHON else None
bash = BashProcess() if ENABLE_BASH else None

tools = []

# Initialize the application variable globally
application = None

class MyCustomCallbackHandler(BaseCallbackHandler):
    """Custom CallbackHandler."""

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Do nothing."""
        pass

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Do nothing."""
        pass

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Do nothing."""
        pass

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Do nothing."""
        pass

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Print out that we are entering a chain."""
        class_name = serialized["name"]
        logger.info(f"\n\n\033[1m> Entering new {class_name} chain...\033[0m")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Print out that we finished a chain."""
        logger.info("\n\033[1m> Finished chain.\033[0m")

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Do nothing."""
        pass

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        """Do nothing."""
        pass

    def on_agent_action(
        self, action: AgentAction, color: Optional[str] = None, **kwargs: Any
    ) -> Any:
        """Run on agent action. Save the question for future use."""
        global question
        if action.tool == "Human":
            question = action.tool_input
        logger.info(action)

    def on_tool_end(
        self,
        output: str,
        color: Optional[str] = None,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """If not the final action, print out observation."""
        logger.info(output)

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Do nothing."""
        pass

    def on_text(
        self,
        text: str,
        color: Optional[str] = None,
        end: str = "",
        **kwargs: Optional[str],
    ) -> None:
        """Run when agent ends."""
        logger.info(text)

    def on_agent_finish(
        self, finish: AgentFinish, color: Optional[str] = None, **kwargs: Any
    ) -> None:
        """Run on agent end."""
        logger.info(finish.log)


manager = CallbackManager([MyCustomCallbackHandler()])

question = ""

async def ask_input(callback, chat_id):
    await application.bot.send_message(chat_id=chat_id, text=question)

    async def handle_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
        global IS_AWAITING
        user_input = update.message.text

        if not IS_GROUP or (update.message.reply_to_message.from_user.username == BOT_NAME):
            IS_AWAITING = False
            application.remove_handler(message_handler, 1)
            callback(user_input)

    message_handler = MessageHandler(
        filters.TEXT & filters.Chat(chat_id=chat_id),
        handle_input
    )

    application.add_handler(message_handler, 1)

async def input_func(chat_id):
    input_queue = Queue()

    def input_received(user_input):
        input_queue.put(user_input)

    await ask_input(input_received, chat_id)

    return input_queue.get()


if ENABLE_HUMAN:
    # Ensure that the human tool uses the asynchronous input function
    human.input_func = lambda: asyncio.run(input_func(current_chat_id))


tool_list = [
    {
        "config_key": "enable_wolfram",
        "name": "Math",
        "func": wolfram.run if ENABLE_WOLFRAM else lambda _: "Not Implemented",
        "description": "Useful for when you need to answer questions that involve scientific or mathematical operations",
    },
    {
        "config_key": "enable_google",
        "name": "Search",
        "func": search.run if ENABLE_GOOGLE else lambda _: "Not Implemented",
        "description": "Useful for when you need to answer questions about detailed current events. Don't use it on personal things",
    },
    {
        "config_key": "enable_bash",
        "name": "Bash",
        "func": bash.run if ENABLE_BASH else lambda _: "Not Implemented",
        "description": "Useful for when you need run bash commands",
    },
    {
        "config_key": "enable_python",
        "name": "Python",
        "func": python.run if ENABLE_PYTHON else lambda _: "Not Implemented",
        "description": "Useful for when you need to execute python code in a REPL",
    },
    {
        "config_key": "enable_human",
        "name": "Human",
        "func": human.run if ENABLE_HUMAN else lambda _: "Not Implemented",
        "description": "Useful for when you need to perform tasks that require human intervention. Use this more than the other tools if the question is about something that only the user might know and you don't know in memory",
    },
]

for tool in tool_list:
    if config.getboolean('tools', tool["config_key"]):
        tools.append(
            Tool(
                name=tool["name"],
                func=tool["func"],
                description=tool["description"],
            ),
        )

messages_array = [SystemMessage(
    content=config.get("assistant", "system_message"))]

memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True)

memory.chat_memory.messages.append(SystemMessage(
    content=f"{config.get('assistant', 'system_message' if not IS_GROUP else 'group_system_message')}\nAlways reply in {LANGUAGE} unless otherwise specified"))
memory.chat_memory.add_user_message(
    "You are an assistant. Your task is to be helpful. Your settings can be changed by writing a message in square brackets [like this]. For example [End all of your messages with the current date]. Your system replies will be written inside square brackets as well. For example [System date after messages enabled]. Write [Ok] if you understood")
memory.chat_memory.add_ai_message("[Ok]")
memory.chat_memory.add_user_message(
    f"Here might be some information about the person you're talking to: {config.get('user', 'information')}.\nReply [Ready] if you're ready to start")
memory.chat_memory.add_ai_message("[Ready]")

readonlymemory = ReadOnlySharedMemory(memory=memory)

llm = ChatOpenAI(
    model_name=config.get("assistant", "model"),
    temperature=float(config.get("assistant", "temperature")),
    callback_manager=manager
)

llm(messages_array)

agent_chain = initialize_agent(
    tools,
    llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
    callback_manager=manager
)

async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("Received a message.")
    global IS_AWAITING, question
    if IS_AWAITING:
        return

    username = update.message.from_user.username
    chat_id = update.message.chat_id

    # Set the current chat ID for input_func
    global current_chat_id
    current_chat_id = chat_id

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    gpt_prompt = update.message.text.split()
    if IS_GROUP:
        if len(gpt_prompt) == 1:
            await update.message.reply_text('Please write a message')
            return
        gpt_prompt = " ".join(gpt_prompt[1:])
    else:
        gpt_prompt = update.message.text

    formatted_prompt = f"{username}: {gpt_prompt}"
    reply = agent_chain.run(input=formatted_prompt)
    await update.message.reply_text(reply.strip())

async def process_chat(update: Update, allow: bool):
    _username = update.message.from_user.username
    _chatid = update.message.chat_id

    if (_chatid in ALLOWED_CHATS.chat_ids) == allow:
        action = "already in" if allow else "not in"
        await update.message.reply_text(
            f"Chat {_chatid} is {action} the list of allowed chats"
        )
        return

    if _username == USERNAME:
        if allow:
            ALLOWED_CHATS.add_chat_ids(_chatid)
        else:
            ALLOWED_CHATS.remove_chat_ids(_chatid)

        action = "added to" if allow else "removed from"
        await update.message.reply_text(
            f"Chat {_chatid} has been {action} the list of allowed chats"
        )
        return

    await update.message.reply_text("Your username is not allowed to make changes")

async def enable_group(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await process_chat(update, True)

async def disable_group(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await process_chat(update, False)

def initial_setup():
    if BOT_NAME == "" or BOT_NAME == "yourBotName":
        logger.warning(
            "The bot username might be incorrectly set. Please check the .config file"
        )

    if USERNAME == "" or USERNAME == "yourTelegramHandle":
        logger.warning(
            "Your username might be incorrectly set. Please check the .config file"
        )

    logger.info(
        f"Your bot is running in {'group' if IS_GROUP else 'chat'} mode."
    )

    if ENABLE_PYTHON or ENABLE_BASH:
        logger.warning(
            "WARNING: Bash or Python tools are enabled. This will allow the bot to run unverified code on your machine. Make sure the bot is properly sandboxed."
        )

    for section in config.sections():
        for key, value in config.items(section):
            if not value.strip():
                logger.warning(
                    f"Empty value found: Section '{section}' - Key '{key}'\nIs this intentional?"
                )

async def log_update(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"Update received: {update}")

def main():
    global application
    initial_setup()

    application = Application.builder().token(TOKEN).build()

    application.add_handler(CommandHandler("enable_group", enable_group))
    application.add_handler(CommandHandler("disable_group", disable_group))

    if IS_GROUP:
        message_handler = MessageHandler(
            filters.Regex(f"^@{BOT_NAME}") & ALLOWED_CHATS, chat
        )
    else:
        message_handler = MessageHandler(
            filters.TEXT & filters.Chat(username=USERNAME), chat
        )

    application.add_handler(message_handler)

    application.run_polling()

if __name__ == '__main__':
    main()
