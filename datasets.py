
import numpy as np
import pandas as pd
import sklearn 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle 
import sys 
import seaborn 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")

with open('../ADNI.pkl', 'rb') as f:
    d = pickle.load(f)


print('Dataset size', d.shape)

initial_keys = list(d.keys())

# Numerical Vs Categorical Data 
numeric_data = d.select_dtypes(include=[np.number])
categorical_data = d.select_dtypes(exclude=[np.number])

print(type(categorical_data))
print(type(numeric_data))


print('Numerical Attributes')
print(len(list(numeric_data.keys()))) 
print('Categorical Attributes')
print(len(list(categorical_data.keys())))

# Nan values in categorical data 

cleaned_cd = categorical_data.dropna()

print('Cleaned Categorical Data',len(list(cleaned_cd.keys()))) # seems no missing categorical information 

# Analysis of categorical data : What do they represent ? 

categorical_keys = list(categorical_data.keys()) 

for ck in categorical_keys :
    print(ck, cleaned_cd[ck].dtype)



# Nan values in numerical data 



   



