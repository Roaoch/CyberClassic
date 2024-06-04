import nltk
import asyncio
import os

from src.gan import GAN
from src.bot.markup import menu_inline, menu_keyboard

from aiogram import Bot, Dispatcher
from aiogram.types import Message
from aiogram.enums.parse_mode import ParseMode
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram import Router, F
from aiogram.filters import Command
from aiogram.client.default import DefaultBotProperties
from dotenv import load_dotenv

nltk.download('punkt')
nltk.download('stopwords')

load_dotenv()

text_generator = GAN(
    min_length=20,
    max_length=40,
    df_path='./dataset.csv',
    false_df_path='./false_dataset.csv',
    machine_df_path='./false_dataset.csv',
    is_train=False,
    n_epochs=12
)
# router = Router()

# TOKEN = os.getenv('bot_key')

# async def main():
#     bot = Bot(
#         token=TOKEN, 
#         default=DefaultBotProperties(parse_mode=ParseMode.HTML)
#     )
#     dp = Dispatcher(storage=MemoryStorage())
#     dp.include_router(router)
#     await bot.delete_webhook(drop_pending_updates=True)
#     await dp.start_polling(
#         bot, 
#         allowed_updates=dp.resolve_used_update_types()
#     )

# @router.message(Command('start'))
# async def start(msg: Message):
#     await msg.answer(
#         'Привет, я нейросеть генерирующий тексты. Я обучен на корпусе текстов Фёдора Михаайловича Достоеевского!\nSpecial thanks:\
#             \n    Фёдор Михаайлович Достоеевский\
#             \nРазработчики:\
#             \n    Горшенин А.К\
#             \n    Мыльников Н.В\
#             \n    Колин А.В\
#             \n    Закиров Р.М\
#             \n    Смирнов И.С\
#             \n    Воробъёв А.И\
#             \nМои потроха:',
#         reply_markup=menu_inline
#     )
#     await msg.answer('Чтобы сгенерировать новое предложение - нажмите на кнопку клавиатуры', reply_markup=menu_keyboard)

# @router.message(F.text == 'Сгенерировать предложение')
# async def text_handler(msg: Message):
#     await msg.answer('Подождите буквально пару секунд')
#     text = text_generator.generate()
#     await msg.answer(f'Достоевский: {text}', reply_markup=menu_keyboard)

# if __name__ == "__main__":
#     asyncio.run(main())