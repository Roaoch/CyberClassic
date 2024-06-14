import warnings
import logging

from src.gan import GAN
from fastapi import FastAPI

warnings.simplefilter("ignore", UserWarning)
logger = logging.getLogger(__name__)

app = FastAPI()

text_generator = GAN(
    min_length=30,
    max_length=50,
    df_path='./dataset.csv',
    false_df_path='./false_dataset.csv',
    machine_df_path='./false_dataset.csv',
    is_train_generator=False,
    is_train_discriminator=False,
    is_train_gan=False,
    n_epochs=8
)

@app.get("/")
def generete():
    return {"text": str(text_generator.generate())}