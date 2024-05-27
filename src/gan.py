import torch
import random

import pandas as pd

from src.encoder import Encoder, EncoderDataset
from src.descriptor import Descriminator
from src.generator import Generator

from transformers import AutoTokenizer, PreTrainedTokenizer
from torch.utils.data import DataLoader
from datasets import Dataset
from typing import List
from tqdm.auto import tqdm

import numpy as np

class GAN(torch.nn.Module):
    def __init__(
            self,
            min_length: int, 
            max_length: int,
            df_path: str,
            false_df_path: str,
            machine_df_path: str,
            is_train: bool,
            n_epochs=3
        ) -> None:
        super().__init__()
        self.min_length = min_length
        self.max_length = max_length
        self.df = pd.read_csv(df_path)
        self.false_df = pd.read_csv(false_df_path)
        self.machine_df = pd.read_csv(machine_df_path)
        self.ds = Dataset.from_pandas(self.df[:2000])

        self.train_dataloader = DataLoader(
            dataset=self.ds,
            shuffle=True,
            batch_size=64
        )

        if is_train:
            print('<--- Train Tokenizer --->')
            self.tokenizer = self._train_tokenizer()
            print('<--- Training Tokenizer end --->')
        else:
            self.tokenizer = AutoTokenizer.from_pretrained('Roaoch/CyberClassic/tokenizer')

        self.encoder = Encoder(
            tokenizer=self.tokenizer,
            is_train=False,# is_train,
            n_epochs=n_epochs,
            dataset=EncoderDataset(
                tokenizer=self.tokenizer,
                df=self.df,
                batch_size=64
            )
        )
        self.descriminator = Descriminator(
            encoder=self.encoder,
            n_epoch=n_epochs,
            is_train=False,# is_train,
            true_df=self.df[:3000],
            false_df=self.false_df,
            batch_size=64
        )
        self.generator = Generator(
            tokenizer=self.tokenizer,
            encoder=self.encoder,
            descriminator=self.descriminator,
            df=self.df[:3000],
            is_train=False,# is_train,
            n_epoch=n_epochs,
            batch_size=64
        )

        if False:
            self.train(n_epoch=20)

    def _train_tokenizer(self) -> PreTrainedTokenizer:
        old_tokenizer = AutoTokenizer.from_pretrained('ai-forever/ruBert-base')
        tokenizer: PreTrainedTokenizer = old_tokenizer.train_new_from_iterator(self.df['text'].values, len(old_tokenizer.vocab))
        tokenizer.save_pretrained('Roaoch/CyberClassic/tokenizer')
        return tokenizer

    def train(self, n_epoch: int):
        print('<--- Train GAN --->')
        lr = 1e-3
        loss_fn = torch.nn.BCELoss()
        optimizer_discriminator = torch.optim.AdamW(self.descriminator.model.parameters())
        scheduler_discriminator = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer_discriminator,
            max_lr=lr,
            epochs=n_epoch,
            steps_per_epoch=len(self.train_dataloader)
        )
        optimizer_generator = torch.optim.AdamW(self.generator.model.parameters())
        scheduler_generator = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer_generator,
            max_lr=lr,
            epochs=n_epoch,
            steps_per_epoch=len(self.train_dataloader)
        )

        progress_bar = tqdm(range(n_epoch * len(self.train_dataloader)))

        for i in range(n_epoch):
            epoch_loss_generator = 0
            epoch_loss_descriminator = 0
            for batch in self.train_dataloader:
                self.descriminator.zero_grad()

                target: torch.Tensor = torch.full((len(batch['text']),), 1.).float()
                output_descriminator = self.descriminator(batch['text']).view(-1)
                loss_descriminator_real = loss_fn(output_descriminator, target)
                loss_descriminator_real.backward()

                noise = torch.randn((len(batch['text']), 768))
                fake = self.generator.model(noise)
                target.fill_(0.)
                output_descriminator = self.descriminator.predict(fake.detach()).view(-1)
                loss_descriminator_fake = loss_fn(output_descriminator, target)
                loss_descriminator_fake.backward()
                epoch_loss_descriminator += (loss_descriminator_fake + loss_descriminator_real).item()

                optimizer_discriminator.step()
                scheduler_discriminator.step()

                self.generator.zero_grad()

                target.fill_(1.)
                output_descriminator = self.descriminator.predict(fake).view(-1)
                loss_generator = loss_fn(output_descriminator, target)
                loss_generator.backward()

                epoch_loss_generator += loss_generator.item()

                optimizer_generator.step()
                scheduler_generator.step()

                progress_bar.update(1)

            tqdm.write(f'Epoch= {i}, Loss Generator= {epoch_loss_generator}, Loss Descriminator= {epoch_loss_descriminator}')

        print('<--- Training GAN end --->')
        self.generator.model.save_pretrained('Roaoch/CyberClassic/generator')
        self.descriminator.model.save_pretrained('Roaoch/CyberClassic/descriminator')



        

