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
# sns.set_theme(style="whitegrid")

def select_baseline_data(df): 
    '''
    Selects for every participant_id the 1st date that we obtained scans 
    df : dataframe with the input data
    returns : baseline_df 
    '''

    baseline_df  = df.groupby('participant_id',as_index=False).head(1)

    # Keep only the H_MUSE_VOLUME* features 
    features = baseline_df.filter(regex=("H_MUSE_Volume_*"))
    
    # include here the biological markers
    # PIB_Status is 100% missing, so we discard it from the feature set 
    biomarkers = ['PTau_CSF','AV45_SUVR','FDG','Tau_CSF','Abeta_CSF']
    # biomarkers = ['PTau_CSF','FDG','Tau_CSF','Abeta_CSF'] # ADNI1 case



    # Extract biological markers 
    PTau_CSF = baseline_df['PTau_CSF']
    # print(PTau_CSF.shape)
    AV45_SUVR = baseline_df["AV45_SUVR"] # remove it in the case of ADNI1 
    # print(AV45_SUVR.shape)
    FDG = baseline_df["FDG"]
    # print(FDG.shape)
    Tau_CSF =  baseline_df["Tau_CSF"]
    # print(Tau_CSF.shape)
    Abeta_CSF =  baseline_df["Abeta_CSF"]
    # print(Abeta_CSF.shape)
 

    # extract the target (Diagnosis)
    diagnosis = baseline_df['Diagnosis']

    result = pd.concat([diagnosis, features, PTau_CSF, AV45_SUVR, FDG, Tau_CSF, Abeta_CSF ], axis=1, join='inner')

    print(result.head())
    
    return result  

def missing_data(df): 
  '''
  Function that calculates the missing percentage for every feature in a dataframe
  '''

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


def zero_imputation(df, feature): 

  values = {feature : 0} 
  df.fillna(value=values)

  return df

def regressedImpute(X_baseImputed, X_miss, computePerFeatureStatistics = False):
  '''
  Returns :
    X_imputed which has mean of the linearly regressed value instead of the missing values and same shape as X_miss.
  if computePerFeatureStatistics is True, also:
    list of Frobenius norms of difference between reconstructions and original data (without missing values) calculated after each imputing each column.
    list of accuracies on test set of Logistic Regression classifier trained on imputed data after each imputing each column.
  '''
  X_imputed = X_baseImputed.copy()
  frobenius_norms =[]
  accuracies =[]

  for i in range(X_imputed.shape[1]):
    # search for the positions of NaNs in each column
    X_column = X_baseImputed[:,i]
    nan_index = [n for n, val in enumerate(np.isnan(X_miss[:,i])) if val] 
    if len(nan_index) == 0:
      continue 
    val_index = [n for n, val in enumerate(np.isnan(X_miss[:,i])) if ~val] 
    train_x = np.delete(X_baseImputed[val_index], i, 1)
    train_y = X_column[val_index]
    predict_x = np.delete(X_baseImputed[nan_index], i, 1)

    # linear regression
    assert ~np.isnan(train_x).any() 
    assert ~np.isnan(train_y).any()


    clf = LinearRegression().fit(X=train_x,y=train_y)
    X_column[nan_index] = clf.predict(predict_x)

    # update X_imputed
    X_imputed[:,i] = X_column

    if computePerFeatureStatistics == True:
        
        clf_log = LogisticRegression(random_state=0).fit(X=X_imputed,y=y_train)
        frobenius_norms.append(LA.norm(X_imputed - X_train))
        # accuracies.append(clf_log.score(X_test, y_test))

  if computePerFeatureStatistics == True:
    return X_imputed, frobenius_norms, accuracies
  else:
    return X_imputed

def mean_imputation(df): 
  '''
  df : dataframe with features 
  returns : imputed_df : dataframe where the NaN values are substituted with the mean of the column. 
  '''

  imputed_df = df.copy()
  keys = list(df.keys())

  for k in keys: 

    if df[k].isnull().values.any() :
    
      assert imputed_df[k].isna().values.any()
      # print(imputed_df[k].head())
      mean_value = df[k].mean() 
      print('mean value', mean_value)
      imputed_df[k].fillna(mean_value, inplace = True) 
      # print(imputed_df[k].head())
      assert ~(imputed_df[k].isna().values.any())

  return imputed_df 
  

def evaluate(prediction, score, groundtruth, algo): 

  '''
  prediction : list with the predictions 
  groundtruth : array with the groundtruth classes 
  score : array/list with scores of the possitive class extracted from the classifier (either decision_function() or predict_proba()) 
  Class 0 : CN
  Class 1 : Dementia 
  '''

  accuracy = accuracy_score(y_true=groundtruth, y_pred=prediction) 
  precision = sklearn.metrics.precision_score(y_true=groundtruth, y_pred=prediction )
  recall = sklearn.metrics.recall_score(y_true=groundtruth, y_pred=prediction)
  class_report = sklearn.metrics.classification_report(y_true=groundtruth, y_pred=prediction, output_dict=True)

  sensitivity = class_report['1']['recall']
  specificity = class_report['0']['recall']


  # ROC_AUC plot 
  n_classes = 2 
  fpr = {}
  tpr = {}
  roc_auc = {'0': [], '1':[]}
  for i in range(n_classes+1):

      fpr[i], tpr[i], _ = sklearn.metrics.roc_curve(groundtruth, score)
      roc_auc[i] = sklearn.metrics.auc(fpr[i], tpr[i])

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
  plt.savefig('/content/drive/My Drive/CIS520_PROJECT/roc_curve_' + algo + '.png')  
  plt.show()


  return accuracy, precision, recall, sensitivity, specificity
  
