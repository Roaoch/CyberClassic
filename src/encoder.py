import torch
import math

import pandas as pd

from transformers import \
    BertForMaskedLM, \
    PreTrainedTokenizer, \
    DataCollatorForWholeWordMask, \
    default_data_collator
from transformers.modeling_outputs import MaskedLMOutput
from datasets import Dataset
from accelerate import Accelerator
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from typing import Callable, Tuple


# device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.empty_cache()


class EncoderDataset():
    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            df: pd.DataFrame,
            batch_size=64
        ) -> None:
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.data = self._prep_data(df=df, batch_size=batch_size)
        self.data_col = DataCollatorForWholeWordMask(tokenizer, mlm_probability=0.15)

    def _get_collactor_func(self) -> Callable:
        def insert_random_mask(batch):
            features = [dict(zip(batch, t)) for t in zip(*batch.values())]
            masked_inputs = self.data_collator(features)
            return {"masked_" + k: v.numpy() for k, v in masked_inputs.items()}
        return insert_random_mask

    def _prep_data(self, df: pd.DataFrame, batch_size=int) -> Tuple[DataLoader, DataLoader]:
        def _dataset_tokenize(e):
            encoded = self.tokenizer(e['text'])
            return {
                'input_ids': encoded.input_ids,
                'attention_mask': encoded.attention_mask
            }
        
        def group_texts(e):
            concatenated_e = {k: sum(e[k], []) for k in e.keys()}
            total_length = len(concatenated_e[list(e.keys())[0]])
            total_length = (total_length // batch_size) * batch_size
            result = {
                k: [t[i : i + batch_size] for i in range(0, total_length, batch_size)]
                for k, t in concatenated_e.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result

        dataset = Dataset.from_pandas(df)
        dataset = dataset.map(
            _dataset_tokenize, 
            remove_columns=['text']
        )
        dataset = dataset.map(
            group_texts, 
            batched=True
        )
        dataset = dataset.train_test_split(test_size=int(len(dataset) * 0.1))

        train_dataloader = DataLoader(
            dataset=dataset['train'],
            shuffle=True,
            batch_size=batch_size,
            collate_fn=default_data_collator
        )
        test_dataloader = DataLoader(
            dataset=dataset['test'],
            shuffle=False,
            batch_size=batch_size,
            collate_fn=default_data_collator
        )

        return (train_dataloader, test_dataloader)


class Encoder(torch.nn.Module):
    def __init__(
            self, 
            tokenizer: PreTrainedTokenizer,
            dataset: EncoderDataset,
            n_epochs: int,
            is_train=False,
            *args, 
            **kwargs
        ) -> None:
        super().__init__(*args, **kwargs)

        self.tokenizer = tokenizer
        self.dataset = dataset
        self.n_epochs = n_epochs
        self.train_dataloader = dataset.data[0]
        self.test_dataloader = dataset.data[1]

        if is_train:
            self.model = self.train(n_epochs)
        else:
            self.model = BertForMaskedLM.from_pretrained('Roaoch/CyberClassic/encoder')

    def forward(self, x: str) -> torch.Tensor:
        tokens = self.tokenizer(x, return_tensors='pt')
        encoded = self.model.bert(**tokens)
        return encoded.last_hidden_state[0][0]
    
    def encode(self, x: str, max_len=40) -> torch.Tensor:
        tokens = self.tokenizer(
            x,
            truncation=True,
            padding=True,
            add_special_tokens=False,
            pad_to_multiple_of=max_len, 
            max_length=max_len, 
            return_tensors='pt'
        )
        encoded = self.model.bert(**tokens)
        return encoded.last_hidden_state[0]

    def train(self, n_epoch: int) -> BertForMaskedLM:
        print('<--- Train Encoder --->')
        lr = 1e-3
        num_update_steps_per_epoch = len(self.train_dataloader)

        model = BertForMaskedLM.from_pretrained('ai-forever/ruBert-base')
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=lr,
            epochs=n_epoch,
            steps_per_epoch=num_update_steps_per_epoch
        )

        progress_bar = tqdm(range(n_epoch * num_update_steps_per_epoch))

        for i in range(n_epoch):
            epoch_loss = 0
            for batch in self.train_dataloader:
                outputs: MaskedLMOutput = model(**batch)
                loss = outputs.loss
                loss.backward()

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

                epoch_loss += loss.item()
            
            with torch.no_grad():
                losses = []
                valid_bar = tqdm(range(len(self.test_dataloader)))
                for batch in self.test_dataloader:
                    outputs = model(**batch)
                    loss = outputs.loss
                    losses.append(loss.item())
                    valid_bar.update(1)

                losses = torch.Tensor(losses)
                perplexity = math.exp(torch.mean(losses))
                tqdm.write(f'Epoch= {i}, Perplexity={perplexity}, Loss= {epoch_loss}')

        model.save_pretrained('Roaoch/CyberClassic/encoder')
        print('<--- Training Encoder end --->')
        return model