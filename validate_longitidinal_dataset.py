import numpy as np
import pandas as pd
import pickle 
import sys 
import seaborn 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")
# from triplet_loss import create_triples_from_set 



with open('longitudinal_dataset.pkl', 'rb') as f:
    d = pickle.load(f)

print(type(d))  

l = d['dataset']

for i, (f,t) in enumerate(l) : 
    print(f.shape, t)
    print(f)
    sys.exit(0)














