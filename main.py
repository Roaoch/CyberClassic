import asyncio
import os
import warnings
import logging
import requests
import json 

from src.bot.markup import menu_inline, menu_keyboard

from aiohttp import web

from aiogram import Bot, Dispatcher
from aiogram.types import Message
from aiogram.enums.parse_mode import ParseMode
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram import Router, F
from aiogram.filters import Command
from aiogram.client.default import DefaultBotProperties
from aiogram.utils.markdown import hbold
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application

from dotenv import load_dotenv

warnings.simplefilter("ignore", UserWarning)
logger = logging.getLogger(__name__)

load_dotenv()

router = Router()

TOKEN = os.getenv('BOT_TOKEN')
WEB_SERVER_HOST = "127.0.0.1"
WEB_SERVER_PORT = 8080
WEBHOOK_PATH = "/webhook"
WEBHOOK_SECRET = "pfoasofh782gru23buif78dvfasfasfv"
BASE_WEBHOOK_URL = "https://cyberclassic.onrender.com"

async def on_startup(bot: Bot) -> None:
    await bot.set_webhook(f"{BASE_WEBHOOK_URL}{WEBHOOK_PATH}", secret_token=WEBHOOK_SECRET)

def main():
    bot = Bot(
        token=TOKEN, 
        default=DefaultBotProperties(parse_mode=ParseMode.HTML)
    )
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(router)

    dp.startup.register(on_startup)
    app = web.Application()
    webhook_requests_handler = SimpleRequestHandler(
        dispatcher=dp,
        bot=bot,
        secret_token=WEBHOOK_SECRET,
    )
    webhook_requests_handler.register(app, path=WEBHOOK_PATH)
    setup_application(app, dp, bot=bot)
    web.run_app(app, host=WEB_SERVER_HOST, port=WEB_SERVER_PORT)
    # await dp.start_polling(bot)

@router.message(Command('start'))
async def start(msg: Message):
    await msg.answer(
        'Привет, я нейросеть генерирующий тексты. Я обучен на корпусе текстов Фёдора Михаайловича Достоеевского!\nSpecial thanks:\
            \n    Фёдор Михаайлович Достоеевский\
            \nРазработчики:\
            \n    Горшенин А.К\
            \n    Мыльников Н.В\
            \n    Колин А.В\
            \n    Закиров Р.М\
            \n    Смирнов И.С\
            \n    Воробъёв А.И\
            \nМои потроха:',
        reply_markup=menu_inline
    )
    await msg.answer('Чтобы сгенерировать новое предложение - нажмите на кнопку клавиатуры', reply_markup=menu_keyboard)

@router.message(F.text == 'Сгенерировать предложение')
async def text_handler(msg: Message):
    await msg.answer('Подождите буквально пару секунд')
    text = json.loads(requests.get('https://roaoch-cyberclassic.hf.space/').text)['text']
    await msg.answer(f'Достоевский: {text}', reply_markup=menu_keyboard)

if __name__ == "__main__":
    main()