import configparser
import logging
import os
import threading
from queue import Queue
from typing import Any, Dict, List, Optional, Union
import openai
from dotenv import load_dotenv
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.callbacks.base import BaseCallbackHandler, CallbackManager
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from langchain.schema import (AgentAction, AgentFinish,
                              LLMResult, SystemMessage)
from langchain.tools.human.tool import HumanInputRun
from langchain.utilities import BashProcess, GoogleSearchAPIWrapper, PythonREPL
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from telegram import ChatAction
from telegram.ext import Filters, MessageHandler, Updater, CommandHandler

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

load_dotenv()
config = configparser.ConfigParser()
config.read('.config')

openai.api_key = os.getenv("OPENAI_API_KEY")

"""This is an ugly way to provide context access to bot calls"""
upd = None
ctx = None
dp = None
question = ""
""""""

TOKEN = os.getenv("TELEGRAM_TOKEN")
IS_GROUP = config.getboolean("system", "group")
BOT_NAME = config.get("system", "botname")
USERNAME = config.get("user", "username")
LANGUAGE = config.get("system", "language")
IS_AWAITING = False
ALLOWED_CHATS = Filters.chat(chat_id=[])


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


def ask_input(callback):
    global dp
    ctx.bot.send_message(chat_id=upd.effective_chat.id,
                         text=question)

    def handle_input(update, _):
        global IS_AWAITING
        user_input = update.message.text

        if not IS_GROUP or (update.message.reply_to_message.from_user.username == BOT_NAME):
            IS_AWAITING = False
            remove_handler()
            callback(user_input)

    message_handler = MessageHandler(Filters.text & Filters.chat(
        chat_id=upd.effective_chat.id), handle_input, run_async=True)

    def remove_handler():
        dp.remove_handler(message_handler)

    dp.add_handler(message_handler)


def input_func():
    input_queue = Queue()

    def input_received(user_input):
        input_queue.put(user_input)

    input_thread = threading.Thread(target=ask_input, args=(input_received,))
    input_thread.start()
    input_thread.join()

    return input_queue.get()


if config.getboolean("tools", "enable_human"):
    human.input_func = input_func


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

        "description": "Useful for when you need to perform tasks that require human intervention.  Use this more than the other tools if the question is about something that only the user might know and you don't know in memory",
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
    content=f"{config.get('assistant', 'system_message' if not IS_GROUP else 'group_system_message')}\nAlways reply in f{LANGUAGE} unless otherwhise specified"))
memory.chat_memory.add_user_message(
    "You are an assistant. Your task is to be helpful. Your settings can be changed writing a message in square brackets [like this]. For example [End all of your messages with the current date]. Your system replies will be written inside square brackets as well. For example [System date after messages enabled]. Write [Ok] if you understood")
memory.chat_memory.add_ai_message("[Ok]")
memory.chat_memory.add_user_message(
    f"Here might be some information about the person you're talking to: {config.get('user', 'information')}.\nReply [Ready] if you're ready to start")
memory.chat_memory.add_ai_message("[Ready]")

readonlymemory = ReadOnlySharedMemory(memory=memory)

llm = ChatOpenAI(model_name=config.get("assistant", "model"),
                 temperature=config.get("assistant", "temperature"), callback_manager=manager)

llm(messages_array)

agent_chain = initialize_agent(
    tools, llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory, callback_manager=manager)


def chat(update, context):
    global upd, ctx
    upd = update
    ctx = context

    if IS_AWAITING:
        return

    username = update.message.from_user.username
    chat_id = update.message.chat_id

    context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    gpt_prompt = update.message.text.split()
    if IS_GROUP:
        if len(gpt_prompt) == 1:
            update.message.reply_text('Please write a message')
            return
        gpt_prompt = " ".join(gpt_prompt[1:])
    else:
        gpt_prompt = update.message

    formatted_prompt = f"{username}: {gpt_prompt}"
    reply = agent_chain.run(input=formatted_prompt)
    update.message.reply_text(reply.strip())


def process_chat(update, allow):
    _username = update.message.from_user.username
    _chatid = update.message.chat_id

    if (_chatid in ALLOWED_CHATS.chat_ids) == allow:
        action = "already in" if allow else "not in"
        update.message.reply_text(
            f"Chat {_chatid} is {action} the list of allowed chats")
        return

    if _username == USERNAME:
        if allow:
            ALLOWED_CHATS.add_chat_ids(_chatid)
        else:
            ALLOWED_CHATS.remove_chat_ids(_chatid)

        action = "added to" if allow else "removed from"
        update.message.reply_text(
            f"Chat {_chatid} has been {action} the list of allowed chats")
        return

    update.message.reply_text("Your username is not allowed to make changes")


def enable_group(update, _):
    process_chat(update, True)


def disable_group(update, _):
    process_chat(update, False)


def initial_setup():
    if BOT_NAME == "" or BOT_NAME == "yourBotName":
        logger.warn(
            "The bot username might be incorrectly set. Please check the .config file")

    if USERNAME == "" or USERNAME == "yourTelegramHandle":
        logger.warn(
            "Your username might be incorrectly set. Please check the .config file")

    logger.info(
        f"Your bot is running in {'group' if IS_GROUP else 'chat'} mode.")

    if config.getboolean("tools", "enable_python") or config.getboolean("tools", "enable_bash"):
        logger.warn(
            "WARNING: Bash or Python tools are enabled. This will allow the bot to run unverified code on your machine. Make sure the bot is proprly sandboxed.")

    for section in config.sections():
        for key, value in config.items(section):
            if not value.strip():
                logger.warn(
                    f"Empty value found: Section '{section}' - Key '{key}'\nIs this intentional?")


def main():
    initial_setup()

    updater = Updater(TOKEN, use_context=True)

    global dp
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("enable_group", enable_group))
    dp.add_handler(CommandHandler("disable_group", disable_group))

    if (IS_GROUP):
        message_handler = MessageHandler(
            Filters.regex(f"^@{BOT_NAME}") & ALLOWED_CHATS, chat, run_async=True)
    else:
        message_handler = MessageHandler(
            Filters.text & Filters.chat(username=USERNAME), chat, run_async=True)

    dp.add_handler(message_handler)

    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    main()
