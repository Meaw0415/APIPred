import torch
import pandas as pd
import numpy as np




class API_Class(torch.utils.data.Dataset):
    def __init__(self,csv):
        self.csv_value = pd.read_csv(csv)
        self.np_value = np.array(self.csv_value)

    def __getitem__(self,index):
        temp_vector = self.np_value
        label = temp_vector[index][0]
        res_arr = temp_vector[index][1:]
        return res_arr,label
    
    def __len__(self):
        return self.csv_value.shape[0]