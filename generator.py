import torch
import unmask_model
import random

from typing import Optional, List

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

    def generate(self) -> str:
        max_index = len(self.unmasker.vocab) - 1
        text_bad_tokens_ids = np.random.randint(
            low=1, 
            high=max_index,
            size=random.randint(self.min_length, self.max_length)
        ).tolist()
        text_raw = ' '.join([self.unmasker.vocab[id] for id in text_bad_tokens_ids])
        text_tokens_ids = self.unmasker.tokenizer.encode(text_raw)

        for _ in range(20):
            print(self.unmasker.tokenizer.decode(text_tokens_ids))
            mask = np.random.binomial(1, 0.3, size=(len(text_tokens_ids),))
            mask_ids = np.where(mask == 1)[0]
            for rand_i in mask_ids:
                text_tokens_ids[rand_i] = self.unmasker.tokenizer.mask_token_id
            text_tokens_ids: List[int] = self.unmasker(text_tokens_ids)

        return self.unmasker.tokenizer.decode(text_tokens_ids)
        

        

