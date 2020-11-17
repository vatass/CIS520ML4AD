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

def select_baseline_data(df): 
    '''
    Selects for every participant_id the 1st date that we obtained scans 
    df : dataframe with the input data
    returns : baseline_df 
    '''

    baseline_df  = df.groupby('participant_id',as_index=False).head(1)

    # Keep only the H_MUSE_VOLUME* features 
    features = baseline_df.filter(regex=("H_MUSE_Volume_*"))

    # extract the target (Diagnosis)
    diagnosis = baseline_df['Diagnosis']

    result = pd.concat([diagnosis, features], axis=1, join='inner')

    print(result.head())

    return result  

def missing_data(df): 
  rows = df.shape[0]
  keys = list(df.keys())
  missing_percentage = [] 
  
  for ck in keys: 
    a = df[ck].isna().sum()/rows 
    missing_percentage.append(a)
    # print(ck,a)
  assert len(missing_percentage) == len(keys)


def age_correction(X): 

  pass

def random_imputation(dataset, feature): 

  possible_imputation_values = dataset['feature'].unique() 

  for row_indexer in range(len(dataset[feature])): 

    if dataset[feature].loc[row_indexer].isnan(): 
       dataset[feature].loc[row_indexer] = random.choice(possible_imputation_values)


  return dataset

  
def evaluate(prediction, score, groundtruth): 

  '''
  prediction : list with the predictions 
  groundtruth : array with the groundtruth classes 
  Class 0 : CN
  Class 1 : Dementia 
  '''

  accuracy = accuracy_score(y_true=groundtruth, y_pred=prediction) 
  precision = sklearn.metrics.precision_score(y_true=groundtruth, y_pred=prediction )
  recall = sklearn.metrics.recall_score(y_true=groundtruth, y_pred=prediction)


  # ROC_AUC plot 
  n_classes = 2 
  fpr = {}
  tpr = {}
  roc_auc = {'0': [], '1':[]}
  for i in range(n_classes+1):

      fpr[i], tpr[i], _ = sklearn.metrics.roc_curve(groundtruth, score)
      roc_auc[i] = sklearn.metrics.auc(fpr[i], tpr[i])


  print('fpr', fpr)
  print('tpr', tpr)

  # Compute micro-average ROC curve and ROC area
  fpr["micro"], tpr["micro"], _ = sklearn.metrics.roc_curve(groundtruth.ravel(), score.ravel())
  roc_auc["micro"] = sklearn.metrics.auc(fpr["micro"], tpr["micro"])

  plt.figure()
  lw = 2
  plt.plot(fpr[2], tpr[2], color='darkorange',
          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
  plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver operating characteristic example')
  plt.legend(loc="lower right")
  plt.show()


  return accuracy, precision, recall 


def select_baseline_data(df): 
    '''
    Selects for every patient_id the 1st date that we obtained scans 
    df : dataframe with the input data
    '''

    # sort the data for each patient 

    df_pp = df.groupby('patient_id')

    print(df_pp) 