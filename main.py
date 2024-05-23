import prep_data
import unmask_model
import generator
import nltk
import asyncio
import os

from aiogram.types import \
    InlineKeyboardButton, \
    InlineKeyboardMarkup, \
    KeyboardButton,\
    ReplyKeyboardMarkup, \
    ReplyKeyboardRemove, \
    Message
from aiogram import Bot, Dispatcher
from aiogram.enums.parse_mode import ParseMode
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram import Router, types, F
from aiogram.filters import Command
from aiogram.client.default import DefaultBotProperties
from dotenv import load_dotenv

nltk.download('punkt')
nltk.download('stopwords')

load_dotenv()

data_prep = prep_data.DataPreparer(
    'output1.txt',
    'output2.txt',
    'output3.txt',
    'output4.txt',
    'output5.txt',
    'output6.txt',
    'output7.txt',
    'output8.txt',
    'output9.txt',
    'Besy.xlsx',
    'dostoevskii21.xlsx',
    'Idiot.xlsx'
)
df = data_prep.get_df()

unmasker = unmask_model.CyberClassicModel(df, is_train=False)
text_generator = generator.Generator(
    20,
    40,
    unmasker
)
router = Router()

TOKEN = os.getenv('bot_key')

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

menu_inline = InlineKeyboardMarkup(inline_keyboard=[
    [
        InlineKeyboardButton(text='GitHub', url='https://github.com/Roaoch/CyberClassic'),
        InlineKeyboardButton(text='HuggingHub', url='https://huggingface.co/Roaoch/CyberClassic')
    ]
])
menu_keyboard = ReplyKeyboardMarkup(
    keyboard=[
        [
            KeyboardButton(text='Сгенерировать предложение')
        ]
    ],
    resize_keyboard=True,
    input_field_placeholder='Нажми кнопку.'
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
    text = text_generator.generate()
    await msg.answer(f'Достоевский: {text}', reply_markup=menu_keyboard)

if __name__ == "__main__":
    asyncio.run(main())