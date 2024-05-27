import torch

import pandas as pd
import numpy as np

from src.encoder import Encoder

from datasets import Dataset, DatasetDict, concatenate_datasets
from transformers import PretrainedConfig, PreTrainedModel
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from tqdm.auto import tqdm
from typing import List


class DescriminatorModelConfig(PretrainedConfig):
    model_type = 'descriminatormodel'
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class DescriminatorModel(PreTrainedModel):
    config_class = DescriminatorModelConfig
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = torch.nn.Sequential(
            torch.nn.Linear(768*40, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1),
            torch.nn.Sigmoid()
        )
    def forward(self, input):
        return self.model(input) 


class Descriminator(torch.nn.Module):
    def __init__(
            self,
            encoder: Encoder,
            n_epoch: int,
            true_df: pd.DataFrame,
            false_df: pd.DataFrame,
            batch_size: int,
            is_train: False,
            *args, 
            **kwargs
        ) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = encoder
        self.n_epoch = n_epoch
        self.dataset = self._get_dataset(true_df=true_df, false_df=false_df)
        self.train_dataloader = DataLoader(
            dataset=self.dataset['train'],
            shuffle=True,
            batch_size=batch_size,
        )
        self.test_dataloader = DataLoader(
            dataset=self.dataset['test'],
            shuffle=False,
            batch_size=batch_size,
        )

        if is_train:
            self.model = DescriminatorModel(DescriminatorModelConfig())
            self.train(self.n_epoch)
        else:
            self.model = DescriminatorModel.from_pretrained('Roaoch/CyberClassic/descriminator')


    def forward(self, x: List[str]):
        encoded = [self.encoder.encode(text).tolist() for text in x]
        encoded = torch.Tensor(encoded).reshape(len(x), 768*40)
        return self.model(encoded)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def train(
        self, 
        n_epoch: int
    ):
        print('<--- Train Descriminator --->')
        lr = 1e-3

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        loss_fn = torch.nn.BCELoss()
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=lr,
            epochs=n_epoch,
            steps_per_epoch=len(self.train_dataloader)
        )

        progress_bar = tqdm(range(n_epoch * len(self.train_dataloader)))

        for i in range(n_epoch):
            epoch_loss = 0
            for batch in self.train_dataloader:
                labels: torch.Tensor = batch['label'].reshape(-1, 1).float()
                outputs = self(batch['text'])
                loss = loss_fn(outputs, labels)
                loss.backward()

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

                epoch_loss += loss.item()
            
            with torch.no_grad():
                metric = []
                valid_bar = tqdm(range(len(self.test_dataloader)))
                for batch in self.test_dataloader:
                    labels: torch.Tensor = batch['label'].reshape(-1, 1).float()
                    outputs = self(batch['text'])
                    metric.append(self._get_true_negative(outputs, labels))
                    valid_bar.update(1)

                tqdm.write(f'Epoch= {i}, True Negative={np.mean(metric)}, Loss= {epoch_loss}')

        print('<--- Training Descriminator end --->')
        self.model.save_pretrained('Roaoch/CyberClassic/descriminator')

    def _get_true_negative(self, input: torch.Tensor, target: torch.Tensor):
        y_pred_class = (input > 0.5).float()
        tn, fp, fn, tp = confusion_matrix(target, y_pred_class).ravel()
        return tn / (tn + fp)

    def _get_dataset(self, true_df: pd.DataFrame, false_df: pd.DataFrame) -> DatasetDict:
        true_ds = Dataset.from_pandas(true_df[:3500])
        true_ds = true_ds.add_column('label', [1.] * len(true_ds))
        false_ds = Dataset.from_pandas(false_df)
        false_ds = false_ds.add_column('label', [0.] * len(false_ds))
        merged_ds: Dataset = concatenate_datasets([true_ds, false_ds])
        return merged_ds.shuffle().train_test_split(test_size=0.1)
        
