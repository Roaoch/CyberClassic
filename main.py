import prep_data
import unmask_model
import generator
import nltk
import telebot
import os

from telebot import types
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
    'output9.txt'
)

df = data_prep.get_df()

unmasker = unmask_model.CyberClassicModel(df)
text_generator = generator.Generator(
    20,
    40,
    unmasker
)

bot = telebot.TeleBot(os.getenv('bot_key'))

@bot.message_handler(commands = ['start'])
def start(message):
    markup = types.InlineKeyboardMarkup()
    btn1 = types.InlineKeyboardButton(text='GitHub', url='https://github.com/Roaoch/CyberClassic')
    btn2 = types.InlineKeyboardButton(text='HuggingHub', url='https://huggingface.co/Roaoch/CyberClassic')
    markup.add(btn1, btn2)
    bot.send_message(message.from_user.id, "Репозитории", reply_markup = markup)

@bot.message_handler(commands= ['text'])
def text_handler(message):
    if message.text == "/generate":
        bot.send_message(message.from_user.id, f'Достоевский: {text_generator.generate()}')