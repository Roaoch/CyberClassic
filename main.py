import asyncio
import os
import warnings
import logging
import requests
import json 

from src.bot.markup import menu_inline, menu_keyboard

from aiogram import Bot, Dispatcher
from aiogram.types import Message
from aiogram.enums.parse_mode import ParseMode
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram import Router, F
from aiogram.filters import Command
from aiogram.client.default import DefaultBotProperties
from dotenv import load_dotenv

warnings.simplefilter("ignore", UserWarning)
logger = logging.getLogger(__name__)

load_dotenv()

router = Router()

TOKEN = os.getenv('BOT_TOKEN')

async def main():
    bot = Bot(
        token=TOKEN, 
        default=DefaultBotProperties(parse_mode=ParseMode.HTML)
    )
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(router)
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(
        bot, 
        allowed_updates=dp.resolve_used_update_types()
    )

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
    logger.info('Model has been prepared')
    asyncio.run(main())