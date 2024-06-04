import torch
import json

import pandas as pd

from src.descriptor import Descriminator
from src.generator import Generator

from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict
from nltk.translate.bleu_score import sentence_bleu
from tqdm.auto import tqdm
from typing import List

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
        self.n_epoch = n_epochs
        self.df = pd.read_csv(df_path)
        self.false_df = pd.read_csv(false_df_path)
        self.machine_df = pd.read_csv(machine_df_path)

        self.ds = self._get_ds(self.df[:1500])
        self.train_dataloader = DataLoader(
            dataset=self.ds['train'],
            shuffle=True,
            batch_size=16,
        )
        self.test_dataloader = DataLoader(
            dataset=self.ds['test'],
            shuffle=False,
            batch_size=16,
        )

        self.tokenizer = AutoTokenizer.from_pretrained('ai-forever/rugpt3small_based_on_gpt2')
        self.generator = Generator(
            tokenizer=self.tokenizer,
            df=self.df[:5000],
            max_length=max_length,
            min_length=min_length,
            is_tain=is_train,
            batch_size=64,
            n_epoch=n_epochs
        )
        self.descriminator = Descriminator(
            encode_tokens=self.generator.encode,
            tokenizer=self.tokenizer,
            n_epoch=12,
            is_train=is_train,
            true_df=self.df[:2300],
            false_df=self.false_df,
            batch_size=16
        )

        if is_train:
            self.train()

    def train(self):
        print('<--- Train GAN --->')
        lr = 1e-3
        loss_fn = torch.nn.BCELoss()
        optimizer_discriminator = torch.optim.AdamW(self.descriminator.model.parameters())
        scheduler_discriminator = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer_discriminator,
            max_lr=lr,
            epochs=self.n_epoch,
            steps_per_epoch=len(self.train_dataloader)
        )
        optimizer_generator = torch.optim.AdamW(self.generator.model.parameters())
        scheduler_generator = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer_generator,
            max_lr=lr,
            epochs=self.n_epoch,
            steps_per_epoch=len(self.train_dataloader)
        )

        progress_bar = tqdm(range(self.n_epoch * len(self.train_dataloader)))

        metrics = {
            'bleu_score': [],
            'model_out': []
        }

        for i in range(self.n_epoch):
            epoch_loss_generator = 0
            epoch_loss_descriminator = 0

            for batch in self.train_dataloader:
                self.descriminator.zero_grad()
                
                input_tokens = self.tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True)
                input_emb = self.generator.encode(input_tokens)
                target: torch.Tensor = torch.full((len(input_emb),), 1.).float()

                output_descriminator = self.descriminator(input_emb).view(-1)
                loss_descriminator_real = loss_fn(output_descriminator, target)
                loss_descriminator_real.backward()

                noise = self.tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True, max_length=3)
                fake_texts = self.generator.generate(noise)
                fake_tokens = self.tokenizer(self.tokenizer.batch_decode(fake_texts), return_tensors='pt', padding=True, truncation=True)
                fake_embs = self.generator.encode(fake_tokens)

                target.fill_(0.)
                output_descriminator = self.descriminator(fake_embs.detach()).view(-1)
                loss_descriminator_fake = loss_fn(output_descriminator, target)
                loss_descriminator_fake.backward()

                epoch_loss_descriminator += (loss_descriminator_fake + loss_descriminator_real).item()

                optimizer_discriminator.step()
                scheduler_discriminator.step()

                self.generator.zero_grad()

                target.fill_(1.)
                output_descriminator = self.descriminator(fake_embs.detach()).view(-1)
                loss_generator = loss_fn(output_descriminator, target)
                loss_generator.backward()

                epoch_loss_generator += loss_generator.item()

                optimizer_generator.step()
                scheduler_generator.step()

                progress_bar.update(1)

            with torch.no_grad():
                model_outs = []
                valid_bar = tqdm(range(len(self.test_dataloader)))
                for batch in self.test_dataloader:
                    tokens = self.tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True, max_length=3)
                    fake = self.generator.generate(tokens)
                    fake_texts = self.tokenizer.batch_decode(fake)
                    bleu_score = np.mean([sentence_bleu(batch['text'][i], fake_texts[i]) for i in range(len(batch))])

                    fake_tokens = self.tokenizer(fake_texts, return_tensors='pt', padding=True, truncation=True)
                    fake_embs = self.generator.encode(fake_tokens)
                    output_descriminator = self.descriminator(fake_embs.detach()).view(-1)
                    model_outs.append(float(torch.mean(output_descriminator)))

                    valid_bar.update(1)

                model_outs = np.mean(model_outs)
                metrics['bleu_score'].append(bleu_score)
                metrics['model_out'].append(model_outs)    

            tqdm.write(f'Epoch= {i}, Loss Generator= {epoch_loss_generator}, Loss Descriminator= {epoch_loss_descriminator}, Model out= {model_outs}, Bleu= {bleu_score}')

        print('<--- Training GAN end --->')
        self.generator.model.save_pretrained('Roaoch/CyberClassic/Gan/generator')
        with open('gan_metrics.json', 'w') as f:
            json.dump(metrics, f)

    def generate(self) -> List[str]:
        # tokens = self.tokenizer([
        #     'Сложно идти в',
        #     'Естественно долго',
        #     'Служить отечеству',
        #     'Холоп',
        #     'Тихо в',
        #     'Сложно идти в',
        #     'Естественно долго',
        #     'Служить отечеству',
        #     'Холоп',
        #     'Тихо в',
        #     'Сложно идти в',
        #     'Естественно долго',
        #     'Служить отечеству',
        #     'Холоп',
        #     'Тихо в',
        #     'Сложно идти в',
        #     'Естественно долго',
        #     'Служить отечеству',
        #     'Холоп',
        #     'Тихо в',
        # ], return_tensors='pt', padding=True, truncation=True)
        # generated = self.generator.generate(tokens)
        # decoded = self.tokenizer.batch_decode(generated)
        # input_tokens = self.tokenizer(decoded, return_tensors='pt', padding=True, truncation=True)
        # input_emb = self.generator.encode(input_tokens)
        # score = self.descriminator(input_emb)
        # print(score)
        pass

    def _get_ds(self, df: pd.DataFrame) -> DatasetDict:
        return Dataset.from_pandas(df=df).train_test_split(0.1)