import pandas as pd
import numpy as np
data = pd.read_csv("Z:/mlpcnnsemi/dataset/Dataset.csv") 
dataarr=np.array(data)
print(dataarr[0])
# data.head()