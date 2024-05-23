import torch
import string
import collections

import pandas as pd
import numpy as np

from transformers import \
    AutoModelForMaskedLM, \
    BertForMaskedLM, \
    Trainer, \
    TrainingArguments, \
    DataCollatorForLanguageModeling
from transformers import AutoTokenizer, pipeline, PreTrainedTokenizer, PreTrainedTokenizerFast
from tqdm import tqdm

from datasets import Dataset, DatasetDict

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from typing import List

class ModelDataset():
    def __init__(
        self, 
        df: pd.DataFrame,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast
    ) -> None:
        self.tokenizer = tokenizer
        self.data = self._prep_data(df=df)
        self.data_col = DataCollatorForLanguageModeling(tokenizer, mlm_probability=0.15)

    def _prep_data(self, df: pd.DataFrame) -> DatasetDict:
        def _dataset_tokenize(e):
            encoded = self.tokenizer(e['text'])
            return {
                'input_ids': encoded.input_ids,
                'attention_mask': encoded.attention_mask,
                'word_ids': encoded.input_ids
            }
        
        def group_texts(e):
            concatenated_e = {k: sum(e[k], []) for k in e.keys()}
            total_length = len(concatenated_e[list(e.keys())[0]])
            total_length = (total_length // 64) * 64
            result = {
                k: [t[i : i + 64] for i in range(0, total_length, 64)]
                for k, t in concatenated_e.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result

        dataset = Dataset.from_pandas(df)
        dataset = dataset.map(_dataset_tokenize, remove_columns=['text'])
        dataset = dataset.map(group_texts, batched=True)
        return dataset.train_test_split(test_size=int(len(dataset) * 0.1))


class CyberClassicModel(torch.nn.Module):
    def __init__(self, data_df: pd.DataFrame, is_train=False) -> None:
        super().__init__()
        self.df = data_df
        self.vocab = self._get_vocab()
        self.vocab_list = list(self.vocab.values())

        if is_train:
            self.train(data_df)

        self.pipline = pipeline("fill-mask", model="Roaoch/CyberClassic")

    def train(self, df: pd.DataFrame) -> None:
        tokenizer = AutoTokenizer.from_pretrained("ai-forever/ruBert-base")
        tokenizer.add_tokens(list(self.vocab.values()))

        base_model: BertForMaskedLM = AutoModelForMaskedLM.from_pretrained("ai-forever/ruBert-base")
        base_model.resize_token_embeddings(len(tokenizer))
        data = ModelDataset(df=df, tokenizer=tokenizer)

        batch_size = 16
        logging_steps = len(data.data["train"]) // batch_size

        training_args = TrainingArguments(
            output_dir=f'./models/mask-fill-fine',
            overwrite_output_dir=True,
            eval_strategy="epoch",
            learning_rate=2e-5,
            weight_decay=0.01,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            fp16=True,
            push_to_hub=True,
            logging_steps=logging_steps,
        )
        trainer = Trainer(
            model=base_model,
            args=training_args,
            train_dataset=data.data["train"],
            eval_dataset=data.data["test"],
            tokenizer=tokenizer,
            data_collator=data.data_col
        )
        trainer.train()
        trainer.save_model('Roaoch/CyberClassic')
        trainer.push_to_hub()

    def forward(self, x: List[int], is_last=False) -> List[int]:
        inputs = np.array(x)
        mask_token_ids = np.where(inputs == self.pipline.tokenizer.mask_token_id)[0]

        for mask in mask_token_ids:
            text = self.pipline.tokenizer.decode(x)
            sugestions = self.pipline(text)
            if isinstance(sugestions[0], list):
                best = list(sorted(sugestions[0], key=lambda e: e['score'], reverse=True))[0]
            else:
                best = sugestions[0]

            best_token = best['token'] \
                if best['score'] >= 0.1 or is_last \
                else self.pipline.tokenizer.convert_tokens_to_ids(str(np.random.choice(self.vocab_list, 1)[0]))
            x[mask] = best_token
            
            tqdm.write(f'Mask id = {mask}')
            tqdm.write(f'   Actual: {self.pipline.tokenizer.convert_ids_to_tokens(best_token)}')
            tqdm.write(f'   Score: {best["score"]:.2f} Token: {best["token_str"]}')

        return x

    def _get_vocab(self):
        res = {}
        stop_words = set(stopwords.words('russian') + list(string.punctuation))
        for sentence in self.df['text'].values:
            words = word_tokenize(sentence, language='russian')
            for word in words:
                if word not in stop_words:
                    if word not in res:
                        res.update({word.lower(): 1})
                    else:
                        res[word] += 1
        freq = list(sorted(res.items(), key=lambda e: e[1], reverse=True))
        freq = freq[:int(len(freq) * 0.4)]
        words = [item[0] for item in freq]
        res = dict(zip(range(0, len(words)), words))
        return res