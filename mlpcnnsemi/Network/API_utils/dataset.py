import torch
import pandas as pd
import pepfeature as pep
import numpy as np

from API_utils.data_process import count_kmers, merge_apt_pro, protein_feature



class API_Class(torch.utils.data.Dataset):
    def __init__(self,xls_path):

        api_dict = {}
        
        df = pd.read_excel(xls_path)
        count = 0
        
        for index,row in df.iterrows():
            count+=1
            temp_dir={
                    "id":-1,
                    "apt_name":-1,
                    "apt_sequence":-1,
                    "protein_name":-1,
                    "protein_sequence":-1,
                    "class":-1
                }
            
            temp_dir["id"] = row["id"]
            temp_dir["apt_name"] = row["apt"]
            temp_dir["apt_sequence"] = row["apt_sequence"]
            temp_dir["protein_name"] = row["protein"]
            temp_dir["protein_sequence"] = row["protein_sequence"]
            
            if(row["class"]=="positive"):
                temp_dir["class"] = 1
            else:
                temp_dir["class"] = 0
            
            api_dict[index] = temp_dir
            
        self.df  = pd.read_excel
        self.dic = api_dict
        self.ids = count
        
    def __getitem__(self,index):
        temp_dir = self.dic[index]
        
        apt_sequence = temp_dir["apt_sequence"]
        pro_sequence = temp_dir["protein_sequence"]
        
        list_apt = count_kmers(apt_sequence)
        list_pro = protein_feature(pro_sequence)
        
        res_list = merge_apt_pro(list_apt,list_pro)
        res_arr  = np.array(res_list)

        label    = temp_dir["class"]
        # print(type(res_arr),type(label))
        return res_arr,label
    
    def __len__(self):
        return self.ids