
import numpy as np
import pandas as pd
import sklearn 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
import pickle 
import sys 
import seaborn 
import matplotlib.pyplot as plt
import seaborn as sns
from auxfunctions import * 
# sns.set_theme(style="whitegrid")
from auxfunctions import select_baseline_data, missing_data

with open('../ADNI.pkl', 'rb') as f:
    d = pickle.load(f)

# cnt = 0
# keys = list(d.keys())
# for k in keys:
#   # print(k)
#   if k.startswith('PTau_CSF'): 
#     cnt+=1 

biomarkers = ['PTau_CSF','AV45_SUVR','FDG','Tau_CSF','Abeta_CSF', 'PIB_Status']
print('Biomarkers in ADNI dataset :' )
print() 
for bio in biomarkers: 
  # print(d[bio].head())
  print('Biomarker', bio, 'Missing Values :',  d[bio].isnull().sum()/len(d[bio]))
print() 

unique_patients = list(d['participant_id'].unique())
print('Unique Patients', len(unique_patients))
print()

unique_diagnosis = list(d['Diagnosis'].unique())
print('Unique Diagnosis', unique_diagnosis)
print()

#### Set Up Baseline Dataset ### 

# Keep All the datapoints where Phase == 'ADNI1'
# df = d[d['Phase']=='ADNI1']
df = d[d['Diagnosis'] !='MCI']


biomarkers = ['PTau_CSF','AV45_SUVR','FDG','Tau_CSF','Abeta_CSF', 'PIB_Status']
print('Biomarkers in ADNI dataset and Diagnosis!=MCI :' )
print() 
for bio in biomarkers: 
  # print(d[bio].head())
  print('Biomarker', bio, 'Missing Values :',  df[bio].isnull().sum()/len(df[bio]))
print() 


unique_patients = list(df['participant_id'].unique())
print('Unique Patients', len(unique_patients))
print()

unique_diagnosis = list(df['Diagnosis'].unique())
print('Unique Diagnosis', unique_diagnosis)
print()


longitudinal_dataset = set_up_longitudinal_data(dataframe=df)

bdf = select_baseline_data(df=df)

print('Baseline df shape', bdf.shape)


bdf.to_pickle("../data/BaselineADNI1.pkl")

# Missing Values in Baseline 
missing_data(df=bdf)

# Extract Stats 
bdf.describe() 
bdf['Diagnosis'].value_counts()


### DATASET 1 ###
'''
Dataset 1 : Consists only the H_MUSE_** neuroimaging features 

'''

print('DATASET 1 : SET UP')
print() 
### Convert Dataframe to Numpy Array 

Y1 = bdf['Diagnosis'].to_numpy() 
X1 = bdf.filter(regex=("H_MUSE_Volume_*"))

print('Features', X1.shape)
print('Target', Y1.shape)
print() 

Y1[Y1=='Dementia'] = 1 
Y1[Y1=='CN'] = 0 
Y1=Y1.astype('int')


## Store Features and Target in .npy fils## 
np.save('../data/features_1.npy', X1)
np.save('../data/target_1.npy',Y1)



### DATASET 2 ###
print('DATASET 2 : SET UP')
print()
'''
Dataset2 : Consists neuroimaging features (as above) augmented with biological markers : 'PTau_CSF','AV45_SUVR','FDG','Tau_CSF','Abeta_CSF' 
It is more challenging dataset due to the high percentage of missing values in the biological markers.

'''
print(bdf.describe())

### Here we do MEAN Imputation of the missing features #### 
bdf_mean_imputed = mean_imputation(df=bdf)
####


Y2 = bdf_mean_imputed['Diagnosis'].to_numpy() 
Xtemp = bdf_mean_imputed.filter(regex=("H_MUSE_Volume_*"))

# Extract biological markers 
Xtemp = bdf_mean_imputed.filter(regex=("H_MUSE_Volume_*"))
PTau_CSF = np.expand_dims(bdf_mean_imputed['PTau_CSF'].to_numpy(),1)
AV45_SUVR = np.expand_dims(bdf_mean_imputed["AV45_SUVR"].to_numpy(),1)
FDG = np.expand_dims(bdf_mean_imputed["FDG"].to_numpy(),1)
Tau_CSF =  np.expand_dims(bdf_mean_imputed["Tau_CSF"].to_numpy(),1)
Abeta_CSF =  np.expand_dims(bdf_mean_imputed["Abeta_CSF"].to_numpy(),1)

 
X2_mean_imputation = np.concatenate((Xtemp, PTau_CSF, AV45_SUVR, FDG, Tau_CSF, Abeta_CSF ), axis=1)
# X2_mean_imputation = np.concatenate((Xtemp, PTau_CSF, FDG, Tau_CSF, Abeta_CSF ), axis=1) # use in ADNI1 case 

assert ~np.isnan(X2_mean_imputation).any()

print('Features', X2_mean_imputation.shape)
print('Target', Y2.shape)
print()

Y2[Y2=='Dementia'] = 1 
Y2[Y2=='CN'] = 0 
Y2=Y2.astype('int')

## Store Features and Target in .npy fils## 
np.save('../data/mean_imputation_features_2.npy', X2_mean_imputation)
np.save('../data/target_2.npy',Y2)

### Dataset 3 ####
print('DATASET 3 : SET UP')
print()
'''
Dataset3 : Same as Dataset 2 but we perform Regression-Imputation on missing features !
'''

Xtemp = bdf.filter(regex=("H_MUSE_Volume_*"))
PTau_CSF = np.expand_dims(bdf['PTau_CSF'].to_numpy(),1)
AV45_SUVR = np.expand_dims(bdf["AV45_SUVR"].to_numpy(),1)
FDG = np.expand_dims(bdf["FDG"].to_numpy(),1)
Tau_CSF =  np.expand_dims(bdf["Tau_CSF"].to_numpy(),1)
Abeta_CSF =  np.expand_dims(bdf["Abeta_CSF"].to_numpy(),1)

X2 = np.concatenate((Xtemp, PTau_CSF, AV45_SUVR, FDG, Tau_CSF, Abeta_CSF ), axis=1)
# X2 = np.concatenate((Xtemp, PTau_CSF, FDG, Tau_CSF, Abeta_CSF ), axis=1) # use in ADNI1 case

# X2 is X_miss 
# X2_mean_imputation is X_baseImputed 

assert ~np.isnan(X2_mean_imputation).any()

X2_regress_imputed = regressedImpute(X_baseImputed=X2_mean_imputation, X_miss=X2, computePerFeatureStatistics = False)

print('Features', X2_regress_imputed.shape)
print('Target', Y2.shape)
print()

## Store Features and Target in .npy fils## 
np.save('../data/regression_imputation_features_2.npy', X2_regress_imputed)
np.save('../data/target_2.npy',Y2)

