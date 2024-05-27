import torch

import pandas as pd

from src.encoder import Encoder
from src.descriptor import Descriminator

from transformers import PreTrainedModel, PretrainedConfig, PreTrainedTokenizer
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


class GeneratorModelConfig(PretrainedConfig):
    model_type = 'generatormodel'
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class GeneratorModel(PreTrainedModel):
    config_class=GeneratorModelConfig
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = torch.nn.Sequential(
            torch.nn.Linear(768, 1536),
            torch.nn.ReLU(),
            torch.nn.Linear(1536, 3072),
            torch.nn.ReLU(),
            torch.nn.Linear(3072, 6144),
            torch.nn.ReLU(),
            torch.nn.Linear(6144, 768*40),
            torch.nn.LeakyReLU()
        )
    def forward(self, input):
        return self.model(input)
    

class Generator(torch.nn.Module):
    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            encoder: Encoder,
            descriminator: Descriminator,
            df: pd.DataFrame,
            n_epoch: int,
            batch_size: int,
            is_train=False,
            *args, 
            **kwargs
        ) -> None:
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.descriminator = descriminator
        self.batch_size = batch_size
        self.train_dataloader = DataLoader(
            dataset=Dataset.from_pandas(df=df[:3000]),
            shuffle=True,
            batch_size=batch_size
        )
        
        if is_train:
            self.model = GeneratorModel(GeneratorModelConfig())
            self.train(n_epoch)
        else:
            self.model = GeneratorModel.from_pretrained('Roaoch/CyberClassic/generator')

    def train(self, n_epoch: int):
        print('<--- Train Generator --->')
        lr = 1e-3

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        loss_fn = torch.nn.MSELoss()
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
                encoded_true = [self.encoder.encode(text).tolist() for text in batch['text']]
                encoded_true = torch.Tensor(encoded_true).reshape(len(batch['text']), 768*40)
                encoded_cls = [self.encoder(text).tolist() for text in batch['text']]
                encoded_cls = torch.Tensor(encoded_cls)
            
                outputs = self.model(encoded_cls)
                loss = loss_fn(outputs, encoded_true)
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

                epoch_loss += loss.item()

            tqdm.write(f'Epoch= {i}, Loss= {epoch_loss}')
            
            # with torch.no_grad():
            #     metric = []
            #     valid_bar = tqdm(range(len(self.test_dataloader)))
            #     for batch in self.test_dataloader:
            #         labels: torch.Tensor = batch['label'].reshape(-1, 1).float()
            #         outputs = self(batch['text'])
            #         metric.append(self._get_true_negative(outputs, labels))
            #         valid_bar.update(1)

            #     tqdm.write(f'Epoch= {i}, True Negative={np.mean(metric)}, Loss= {epoch_loss}')

        print('<--- Training Generator end --->')
        self.model.save_pretrained('Roaoch/CyberClassic/generator')