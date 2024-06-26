import uuid
import torch
import json

import pandas as pd

from src.descriptor import Descriminator
from src.generator import Generator

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from rouge_score import rouge_scorer
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict
from nltk.translate.bleu_score import SmoothingFunction
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
            is_train_generator: bool,
            is_train_discriminator: bool,
            is_train_gan: bool,
            n_epochs=3
        ) -> None:
        super().__init__()
        self.min_length = min_length
        self.max_length = max_length
        self.n_epoch = n_epochs
        self.df = pd.read_csv(df_path)
        self.false_df = pd.read_csv(false_df_path)

        self.bleu_smoothing = SmoothingFunction().method7
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'])

        self.ds = self._get_ds(self.df[:1500])
        self.train_dataloader = DataLoader(
            dataset=self.ds['train'],
            shuffle=False,
            batch_size=16,
        )
        self.test_dataloader = DataLoader(
            dataset=self.ds['test'],
            shuffle=False,
            batch_size=16,
        )

        self.tokenizer = AutoTokenizer.from_pretrained('Roaoch/CyberClassic-Generator')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.generator = Generator(
            tokenizer=self.tokenizer,
            df=self.df,
            max_length=max_length,
            min_length=min_length,
            is_tain=is_train_generator,
            batch_size=64,
            n_epoch=10
        )
        self.descriminator = Descriminator(
            encode_tokens=self.generator.encode,
            tokenizer=self.tokenizer,
            n_epoch=12,
            is_train=is_train_discriminator,
            true_df=self.df[:5700],
            false_df=self.false_df,
            batch_size=10
        )

        if is_train_gan:
            test_generation = {
                'befor_gan': [],
                'after_gan': []
            }
            test_generation['befor_gan'] = self.test_generate()
            self.train()
            test_generation['after_gan'] = self.test_generate()
            with open('test_generation.json', 'w') as f:
                json.dump(test_generation, f)

    def train(self):
        def get_score(text: str) -> float:
            tokens = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
            embedings = self.generator.encode(**tokens)
            res = self.descriminator(embedings)
            return res * 10
        
        new_model = AutoModelForCausalLMWithValueHead.from_pretrained('Roaoch/CyberClassic/generator')
        new_model_ref = AutoModelForCausalLMWithValueHead.from_pretrained('Roaoch/CyberClassic/generator')

        ppo_config = {
            "mini_batch_size": 1, 
            "batch_size": 20
        }
        config = PPOConfig(**ppo_config)
        ppo_trainer = PPOTrainer(config, new_model, new_model_ref, self.tokenizer)

        query_txts = [
            'Сложно идти в',
            'Естественно долго',
            'Служить отечеству',
            'Холоп',
            'Тихо в',
            'Сложно идти в',
            'Естественно долго',
            'Служить отечеству',
            'Холоп',
            'Тихо в',
            'Сложно идти в',
            'Естественно долго',
            'Служить отечеству',
            'Холоп',
            'Тихо в',
            'Сложно идти в',
            'Естественно долго',
            'Служить отечеству',
            'Холоп',
            'Тихо в',
        ]
        query_tensor = self.tokenizer(
            query_txts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        )['input_ids'].to(new_model.pretrained_model.device)
        query_tensor = [tens for tens in query_tensor]

        generation_kwargs = {
            'max_new_tokens': self.max_length,
            'min_new_tokens': self.min_length,
            'top_k': 0.,
            'top_p': 1.,
            'do_sample': True,
            "pad_token_id": self.tokenizer.eos_token_id
        }

        progress_bar = tqdm(range(100 * 1))
        metrics = {
            'loss': [],
            'reward': [],
        }

        for i in range(100):
            response_tensor = ppo_trainer.generate(query_tensor, return_prompt=False, **generation_kwargs)
            response_txt = self.tokenizer.batch_decode(response_tensor, skip_special_tokens=True)
            reward = get_score(response_txt).to(new_model.pretrained_model.device)
            reward = [rew for rew in reward]
            train_stats = ppo_trainer.step(query_tensor, response_tensor, reward)

            metrics['reward'].append(train_stats['ppo/mean_scores'])
            metrics['loss'].append(train_stats['ppo/loss/value'])
            tqdm.write(f'Epoch= {i} Reward = {train_stats["ppo/mean_scores"]}, Loss = {train_stats["ppo/loss/value"]}')
            progress_bar.update(1)

        new_model.save_pretrained('Roaoch/CyberClassic/RL/generator')
        self.generator.model = AutoModelForCausalLM.from_pretrained('Roaoch/CyberClassic/RL/generator')

        with open('rl_metrics.json', 'w') as f:
            json.dump(metrics, f)

    def test_generate(self) -> List[str]:
        torch.manual_seed(25)
        print('<--- TEST GENERATION --->')
        tokens = self.tokenizer([
            'Сложно идти в',
            'Естественно долго',
            'Служить отечеству',
            'Холоп',
            'Тихо в',
            'Сложно идти в',
            'Естественно долго',
            'Служить отечеству',
            'Холоп',
            'Тихо в',
            'Сложно идти в',
            'Естественно долго',
            'Служить отечеству',
            'Холоп',
            'Тихо в',
            'Сложно идти в',
            'Естественно долго',
            'Служить отечеству',
            'Холоп',
            'Тихо в',
        ], return_tensors='pt', padding=True, truncation=True)
        generated = self.generator.generate(tokens)
        input_emb = self.generator.encode(input_ids=generated, attention_mask=torch.full(generated.size(), 1))

        decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        score = torch.mean(self.descriminator(input_emb))

        to_print = "\n".join(decoded)
        print(f'Mean score = {score}')
        print(f'Generated:\n{to_print}')
        print('<--- TEST GENERATION end --->')
        return decoded

    def generate(self) -> str:
        texts = self.df['text'].values[np.random.randint(0, len(self.df), 4)]
        trimed_texts = [' '.join(txt.split(' ')[:3]) for txt in texts]
        tokens = self.tokenizer(trimed_texts, return_tensors='pt', padding=True, truncation=True)
        generated = self.generator.generate(tokens)

        input_emb = self.generator.encode(input_ids=generated, attention_mask=torch.full(generated.size(), 1))
        score = self.descriminator(input_emb)
        score = torch.abs(score - 0.889)
        index = int(torch.argmin(score))

        decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)

        return decoded[index]


    def _get_ds(self, df: pd.DataFrame) -> DatasetDict:
        return Dataset.from_pandas(df=df).train_test_split(0.1)