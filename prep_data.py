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
            if '.txt' in link:
                with open(f'./data/{link}', encoding='utf8') as text:
                    temp['text'].extend(text.readlines())
            elif '.xlsx' in link:
                df = pd.read_excel(f'./data/{link}', names=['text'])
                temp['text'].extend(df['text'].values)
        return pd.DataFrame(temp)