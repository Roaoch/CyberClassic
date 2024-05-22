import prep_data
import unmask_model
import generator
import torch
import nltk

from transformers import AutoModelForMaskedLM, AutoTokenizer

nltk.download('punkt')
nltk.download('stopwords')

data_prep = prep_data.DataPreparer(
    'output1.txt',
    'output2.txt',
    'output3.txt',
    'output4.txt',
    'output5.txt',
    'output6.txt',
    'output7.txt',
    'output8.txt',
    'output9.txt'
)

df = data_prep.get_df()

unmasker = unmask_model.CyberClassicModel(df)
text_generator = generator.Generator(
    20,
    40,
    unmasker
)

print(text_generator.generate())