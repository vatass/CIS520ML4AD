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
from sklearn.metrics import accuracy_score, plot_roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize
import matplotlib as mpl 


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
  

def plot_cv_indices(cv, X, y, group, ax, n_splits, lw=10):
    """Create a sample plot for indices of a cross-validation object."""

    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y, groups=group)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(range(len(indices)), [ii + .5] * len(indices),
                   c=indices, marker='_', lw=lw, cmap=cmap_cv,
                   vmin=-.2, vmax=1.2)

    # Plot the data classes and groups at the end
    ax.scatter(range(len(X)), [ii + 1.5] * len(X),
               c=y, marker='_', lw=lw, cmap=cmap_data)

    ax.scatter(range(len(X)), [ii + 2.5] * len(X),
               c=group, marker='_', lw=lw, cmap=cmap_data)

    # Formatting
    yticklabels = list(range(n_splits)) + ['class', 'group']
    ax.set(yticks=np.arange(n_splits+2) + .5, yticklabels=yticklabels,
           xlabel='Sample index', ylabel="CV iteration",
           ylim=[n_splits+2.2, -.2], xlim=[0, 100])
    ax.set_title('{}'.format(type(cv).__name__), fontsize=15)
    return ax

def make_ellipses(gmm, ax, colors):
  for n, color in enumerate(colors):
      if gmm.covariance_type == 'full':
          covariances = gmm.covariances_[n][:2, :2]
      elif gmm.covariance_type == 'tied':
          covariances = gmm.covariances_[:2, :2]
      elif gmm.covariance_type == 'diag':
          covariances = np.diag(gmm.covariances_[n][:2])
      elif gmm.covariance_type == 'spherical':
          covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
      v, w = np.linalg.eigh(covariances)
      u = w[0] / np.linalg.norm(w[0])
      angle = np.arctan2(u[1], u[0])
      angle = 180 * angle / np.pi  # convert to degrees
      v = 2. * np.sqrt(2.) * np.sqrt(v)
      ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
                                180 + angle, color=color)
      ell.set_clip_box(ax.bbox)
      ell.set_alpha(0.5)
      ax.add_artist(ell)
      ax.set_aspect('equal', 'datalim')

def multiclass_evaluate(classifier, X_test, Y_test, prediction, score, groundtruth, n_classes,algo): 

  accuracy = accuracy_score(y_true=groundtruth, y_pred=prediction) 
  class_report = sklearn.metrics.classification_report(y_true=groundtruth, y_pred=prediction, output_dict=True)

  groundtruth = label_binarize(groundtruth, classes=[0,1,2])

  # Compute ROC curve and ROC area for each class
  fpr = dict()
  tpr = dict()
  roc_auc = dict()
  for i in range(n_classes):
      fpr[i], tpr[i], _ = sklearn.metrics.roc_curve(groundtruth[:, i], score[:, i])
      roc_auc[i] = sklearn.metrics.auc(fpr[i], tpr[i])

  # Plot of a ROC curve for a specific class
  for i in range(n_classes):
      plt.figure()
      plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
      plt.plot([0, 1], [0, 1], 'k--')
      plt.xlim([0.0, 1.0])
      plt.ylim([0.0, 1.05])
      plt.xlabel('False Positive Rate')
      plt.ylabel('True Positive Rate')
      plt.title('Receiver operating characteristic example')
      plt.legend(loc="lower right")
      plt.savefig('../plots/roc_curve_' + algo + '.png')

  return accuracy, class_report



def evaluate(classifier, X_test, Y_test, prediction, score, groundtruth, n_classes,algo): 

  '''
  prediction : list with the predictions 
  groundtruth : array with the groundtruth classes 
  score : array/list with scores of the possitive class extracted from the classifier (either decision_function() or predict_proba()) 
  Class 0 : CN
  Class 1 : Dementia 
  '''

  accuracy = accuracy_score(y_true=groundtruth, y_pred=prediction) 
  precision = sklearn.metrics.precision_score(y_true=groundtruth, y_pred=prediction,  average='weighted')
  recall = sklearn.metrics.recall_score(y_true=groundtruth, y_pred=prediction,  average='weighted')
  class_report = sklearn.metrics.classification_report(y_true=groundtruth, y_pred=prediction, output_dict=True)

  sensitivity = class_report['1']['recall']
  specificity = class_report['0']['recall']


  # svc_disp = plot_roc_curve(classifier, X_test, Y_test)
  # plt.savefig('roc_curve_ontest'+algo +'.png')
  # plt.show()

  # ROC_AUC plot 
  
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
  plt.savefig('../plots/final_results/baseline_binary_class/roc_curve_' + algo + '.png')  
  # plt.show()
  

  return accuracy, precision, recall, sensitivity, specificity
  
def cross_validation_roc_auc(classifier,X_train, Y_train, algo): 

  cv = StratifiedKFold(n_splits=6)

  tprs = []
  aucs = []
  mean_fpr = np.linspace(0, 1, 100)

  fig, ax = plt.subplots()
  for i, (train, test) in enumerate(cv.split(X_train, Y_train)):
      classifier.fit(X_train[train], Y_train[train])
      viz = plot_roc_curve(classifier, X_train[test], Y_train[test],
                          name='ROC fold {}'.format(i),
                          alpha=0.3, lw=1, ax=ax)
      interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
      interp_tpr[0] = 0.0
      tprs.append(interp_tpr)
      aucs.append(viz.roc_auc)

  ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
          label='Chance', alpha=.8)

  mean_tpr = np.mean(tprs, axis=0)
  mean_tpr[-1] = 1.0
  mean_auc = auc(mean_fpr, mean_tpr)
  std_auc = np.std(aucs)
  ax.plot(mean_fpr, mean_tpr, color='b',
          label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
          lw=2, alpha=.8)

  std_tpr = np.std(tprs, axis=0)
  tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
  tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
  ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                  label=r'$\pm$ 1 std. dev.')

  ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
        title="Receiver operating characteristic example")
  ax.legend(loc="lower right")
  plt.savefig('../plots/roc_auc_cross_val_'+algo+'.png')
  # plt.show()

def plot_uncertainty(groundtruth, uncertainty, path): 

  print(groundtruth.shape, uncertainty.shape)
  plt.figure() 
  sns.boxplot(x=groundtruth, y=uncertainty)
  plt.savefig(path + '.png')
  # plt.show() 


def plot_scores_for_all_algos(picklefiles,names):
  '''
  picklefiles : list of pickle files
  names: list of the plot name to be saved 
  Plot function for evaluation of algorithms in different datasets
  '''

  for picklefile,name in zip(picklefiles, names):

    objects = []
    with (open(picklefile, "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break

    # PLOT SENSITIVITY, SPECIFICITY FOR ALL ALGOS
    plt.figure(figsize=(12,10))
    data = pd.DataFrame(data=objects[0])
    sns.scatterplot("algorithm", "value", hue='metric', data=data)
    plt.title('Evaluation Metrics for ML Algorithms')
    plt.xticks(rotation=45)
    plt.savefig("../plots/"+ name +'.png')
    # plt.show()

def visualize_metrics_for_all_datasets(picklefiles, names):
  
  for i, (picklefile,name) in enumerate(zip(picklefiles, names)):
    
    objects = []
    with (open(picklefile, "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break
    print(objects[0]['algorithm'])
    print(objects[0]['accuracy'])
    print(objects[0]['sensitivity'])
    print(objects[0]['specificity'])
 

    
    continue

    if i == 0: 
      total_scores = {} 
      for key, value in objects[0].items() : 
        total_scores[key] = []
        total_scores[key].extend(value) 
    else :
    
      for key, value in objects[0].items(): 
        total_scores[key].extend(value) 


  data = pd.DataFrame(data=total_scores)

  data1 = data[(data['metric'] != 'precision')] # | data['metric'] == 'specificity'] 
  data2 = data1[data1['algorithm'] != 'KMEANS']
  # data3 = data1[data2['algorithm'] != 'KMEANS']

  sns.relplot(x="algorithm", y="value", hue="metric", style="dataset", data=data2)
  plt.title('Comparison of ML Algorithms performance across datasets')
  plt.xticks(rotation=25)
  plt.savefig("../plots/alldatasets_ml_performance_comparison.png")
  plt.show()




if __name__ == "__main__" : 


  picklefiles= ['sep_scores_dataset1.pickle', 'sep_scores_dataset2.pickle','sep_scores_dataset3.pickle']
  names = ['all_algo_metrics_dataset1', 'all_algo_metrics_dataset2', 'all_algo_metrics_dataset3']
  # plot_scores_for_all_algos(picklefiles=picklefiles, names=names)
  visualize_metrics_for_all_datasets(picklefiles=picklefiles, names=names) 






