import torch
import unmask_model
import random

from typing import Optional, List
from tqdm import tqdm, trange

import numpy as np

class Generator():
    def __init__(
            self,
            min_length: int, 
            max_length: int,
            umasker: Optional[unmask_model.CyberClassicModel]
        ) -> None:
        self.min_length = min_length
        self.max_length = max_length
        self.unmasker: unmask_model.CyberClassicModel = unmask_model.CyberClassicModel() if unmask_model is None else umasker

    def generate(self, n_epochs=10) -> str:
        max_index = len(self.unmasker.vocab) - 1
        text_bad_tokens_ids = np.random.randint(
            low=1, 
            high=max_index,
            size=random.randint(self.min_length, self.max_length)
        ).tolist()
        text_raw = ' '.join([self.unmasker.vocab[id] for id in text_bad_tokens_ids])
        text_tokens_ids = self.unmasker.pipline.tokenizer.encode(text_raw, truncation=True, add_special_tokens=False)

        for i in trange(n_epochs):
            mask = np.random.binomial(1, 0.3, size=(len(text_tokens_ids),))

            tqdm.write(f'Epoch {i}')
            tqdm.write(f'Text: «{self.unmasker.pipline.tokenizer.decode(text_tokens_ids, clean_up_tokenization_spaces=True)}»')
            tqdm.write(f'Mask: «{mask}»')

            mask_ids = np.where(mask == 1)[0]
            for rand_i in mask_ids:
                text_tokens_ids[rand_i] = self.unmasker.pipline.tokenizer.mask_token_id
            text_tokens_ids: List[int] = self.unmasker(text_tokens_ids, is_last=n_epochs==i+1)

            tqdm.write('')
        res = self.unmasker.pipline.tokenizer.decode(text_tokens_ids, clean_up_tokenization_spaces=True)
        print(f'Result: \n«{res}»')
        return res
        

        

