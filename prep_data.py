import pandas as pd
import numpy as np

class DataPreparer:
    def __init__(self, *urls: str) -> None:
        self.links_to_texts = list(urls)
    
    def get_df(self) -> pd.DataFrame:
        temp = {
            'text': []
        }
        for link in self.links_to_texts:
            with open(f'./data/{link}', encoding='utf8') as text:
                temp['text'].extend(text.readlines())
        return pd.DataFrame(temp)