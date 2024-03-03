# Class to take a data set as input and provide a summary of the features and contents of the dataset
import pandas as pd
from datasets import load_dataset
from datasets import load_dataset_builder
from datasets import get_dataset_split_names
import numpy as np


class DatasetInspector:
    def __init__(self, np_array_dataset):
        self.dataset = np_array_dataset

    def load_dataset(self, dataset_path, dataset_type):
        if dataset_type == 'csv':
            self.dataset = pd.read_csv(dataset_path)
            # self.dataset = self.dataset.to_numpy()
        elif dataset_type == 'json':
            self.dataset = np.array(dataset_path)
        elif dataset_type == 'huggingface':
            self.dataset = load_dataset(dataset_path)


    def summarize_dataset(self):
        print(f'Dataset Shape: {self.dataset.shape}')
        

# d = DatasetInspector('pdjewell/sommeli_ai', dataset_type='huggingface')
# d.summarize_dataset()