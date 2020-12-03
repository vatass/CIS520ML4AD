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
from numpy import linalg as LA
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import metrics
import random
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from auxfunctions import *


def KNN(X_train, Y_train, X_val, Y_val, X_test, Y_test, dataset): 
  print('APPLY KNN ...')
  k_values = range(1, 30, 2)
  f1scores = [] 
  for k in range(1, 30, 2):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, Y_train)
    prediction = model.predict(X_val)

    score = sklearn.metrics.f1_score(Y_val, prediction) 

    f1scores.append(score)


  plt.plot(k_values, f1scores, color='g')
  plt.xlabel("K values")
  plt.ylabel("Validation F1 score")
  plt.savefig('../plots/'+dataset +'_knn_kplot.png')
  # plt.show()

  model = KNeighborsClassifier(n_neighbors=20)
  model.fit(X_train, Y_train)
 
  predictions = model.predict(X_test)
  scores = model.predict_proba(X_test)
  acc, prec, rec, sens, spec = evaluate(model, X_test, Y_test, np.array(predictions), np.array(scores[:,1]), np.array(Y_test), 'knn') 
  print('Test Accuracy, Precision, Recall', acc, prec, rec)
  print()

  return acc, prec, rec, sens, spec 

def SVM(X_train, Y_train, X_val, Y_val, X_test, Y_test, dataset):
  print('APPLY SVM ...')

  ### Grid Search to some parameters ####
  params_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

  svm_model = GridSearchCV(SVC(), params_grid, cv=5)
  svm_model.fit(X_train_scaled, Y_train)

  # View the accuracy score
  print('Best score for training data:', svm_model.best_score_,"\n") 

  # View the best parameters for the model found using grid search
  print('Best C:',svm_model.best_estimator_.C,"\n") 
  print('Best Kernel:',svm_model.best_estimator_.kernel,"\n")
  print('Best Gamma:',svm_model.best_estimator_.gamma,"\n")

  final_model = svm_model.best_estimator_

  predictions = final_model.predict(X_test)
  scores = final_model.decision_function(X_test)

  acc, prec, rec, sens, spec = evaluate(final_model, X_test, Y_test, np.array(predictions), np.array(scores), np.array(Y_test), 'svm') 
  print('Test Accuracy, Precision, Recall', acc, prec, rec)
  print()

  #####

  '''
  OLD CODE - See if new works and then delete 

  clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
  crossval_score = cross_validation_roc_auc(classifier=clf, X_train=X_train, Y_train=Y_train, algo='svm')

  # scores in cross-validation 
  scores = cross_val_score(clf, X_train, Y_train, cv=50)

  # mean cross val score
  print('Mean Cross-Val score', np.mean(scores))
  data = {'cross': range(50), 'cv_scores' : scores} 
  sns.lineplot(x='cross', y='cv_scores', data=data)
  plt.savefig('../plots/cross_val_score_svm.png')


  clf.fit(X_train, Y_train)
  predictions = clf.predict(X_test)
  scores = clf.decision_function(X_test)

  acc, prec, rec, sens, spec = evaluate(clf, X_test, Y_test, np.array(predictions), np.array(scores), np.array(Y_test), 'svm') 
  print('Test Accuracy, Precision, Recall', acc, prec, rec)
  print()

  '''

  return acc, prec, rec, sens, spec



def Decision_Tree(X_train, Y_train, X_val, Y_val, X_test, Y_test, dataset):
  print('APPLY DECISION TREE CLASSIFIER ...')

  params = {'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4]}
  grid_search_cv = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1, verbose=1, cv=3)

  grid_search_cv.fit(X_train, Y_train)

  grid_search_cv.best_estimator_

  y_pred = grid_search_cv.predict(X_test)
  accuracy_score(Y_test, y_pred)

  predictions = grid_search_cv.predict(X_test)
  scores = grid_search_cv.predict_proba(X_test)
  acc, prec, rec, sens,spec = evaluate(grid_search_cv, X_test, Y_test, np.array(predictions), np.array(scores[:,1]), np.array(Y_test), 'decision_tree') 
  print('Test Accuracy, Precision, Recall', acc, prec, rec)
  print()

  return acc, prec, rec, sens, spec

def Logistic_Regression(X_train, Y_train, X_val, Y_val, X_test, Y_test, dataset): 
  print('APPLY LOGISTIC REGRESSION ...')

  clf = make_pipeline(StandardScaler(),LogisticRegression(random_state=0))

  logisticregression_score = cross_validation_roc_auc(classifier=clf, X_train=X_train, Y_train=Y_train, algo='logistic_regression')

  # scores in cross-validation 
  scores = cross_val_score(clf, X_train, Y_train, cv=50)

  # mean cross val score
  print('Mean Cross-Val score', np.mean(scores))
  data = {'cross': range(50), 'cv_scores' : scores} 
  sns.lineplot(x='cross', y='cv_scores', data=data)
  plt.savefig('../plots/cross_val_score_logistic_regression.png')

  clf.fit(X_train, Y_train)
  valscore = clf.score(X_val, Y_val)
  # print(valscore) 
  testscore = clf.score(X_test, Y_test)
  testscore

  predictions = clf.predict(X_test)
  scores = clf.predict_proba(X_test)
  acc, prec, rec, sens, spec = evaluate(clf, X_test,Y_test,np.array(predictions), np.array(scores[:,1]), np.array(Y_test), 'logistic') 
  print('Test Accuracy, Precision, Recall', acc, prec, rec)
  print()

  return acc, prec, rec, sens, spec

def AdaBoost(X_train, Y_train, X_val, Y_val, X_test, Y_test, dataset): 
  print('APPLY ADABOOST ...')

  # create a list of all possible depth values
  depths = [1,3,5,8]
  models = []
  train_accuracy = []
  val_accuracy = []
  # create a list of models 
  for depth in depths:
    model = AdaBoostClassifier(
      DecisionTreeClassifier(max_depth=depth), n_estimators=1,
      algorithm="SAMME.R", learning_rate=0.2, random_state=42)
    model.fit(X_train, Y_train)
    models.append(model)
    # evaluate model performance
    train_score = model.score(X_train, Y_train)
    val_score = model.score(X_val, Y_val)
    train_accuracy.append(train_score)
    val_accuracy.append(val_score)
    # print('Depth: {} Train Accuracy: {} Test Accuracy: {}'.format(depth, train_score, val_score))

  # predictions = model.predict(X_test)
  # scores = model.predict_proba(X_test)
  # acc, prec, rec = evaluate(np.array(predictions), np.array(scores[:,1]), np.array(Y_test), 'adaboost') 
  # plot the train and test accuracy
  plt.plot(depths, train_accuracy, '-')
  plt.plot(depths, val_accuracy, '--')
  plt.title('The accuracy of different max_depth')
  plt.savefig('../plots/'+dataset + '_accuracy_depth_adaboost.png')
  # plt.show()

  best_model = models[np.argmax(val_accuracy)]
  best_model.score(X_test, Y_test)

  predictions = best_model.predict(X_test)
  scores = best_model.predict_proba(X_test)
  acc, prec, rec, sens,spec = evaluate(best_model, X_test, Y_test, np.array(predictions), np.array(scores[:,1]), np.array(Y_test), 'adaboost') 
  print('Test Accuracy, Precision, Recall', acc, prec, rec)
  print()


  return acc, prec, rec, sens, spec

def MLP(X_train, Y_train, X_val, Y_val, X_test, Y_test, dataset): 
  print('APPLY MULTILAYER PERCEPTRON ...')

  clf = make_pipeline(StandardScaler(),MLPClassifier(random_state=1, max_iter=300))

  mlp_score = cross_validation_roc_auc(classifier=clf, X_train=X_train, Y_train=Y_train, algo='MLP')

  # scores in cross-validation 
  scores = cross_val_score(clf, X_train, Y_train, cv=50)

  # mean cross val score
  print('Mean Cross-Val score', np.mean(scores))
  data = {'cross': range(50), 'cv_scores' : scores} 
  sns.lineplot(x='cross', y='cv_scores', data=data)
  plt.savefig('../plots/cross_val_score_mlp.png')


  clf.fit(X_train, Y_train)
  predictions = clf.predict(X_test)
  scores = clf.predict_proba(X_test)

  mean_train_accuracy = clf.score(X_train, Y_train)  
  mean_val_accuracy = clf.score(X_val, Y_val)
  mean_test_accuracy = clf.score(X_test, Y_test)
  
  Y_test = list(Y_test) 

  assert len(predictions) == len(scores)
  assert len(scores) == len(Y_test)

  acc, prec, rec, sens, spec = evaluate(clf, X_test, Y_test, np.array(predictions), np.array(scores[:,1]), np.array(Y_test), 'mlp') 
  print('Test Accuracy, Precision, Recall', acc, prec, rec)
  print()

  return acc, prec, rec, sens, spec 


print('RUN SUCCESSFULLY!')


def apply_ML(X_train, Y_train, X_val, Y_val, X_test, Y_test, dataset): 
  ''' 
  X_train, Y_train, X_val, Y_val, X_test, Y_test : sets 
  produces evaluation plots, cumulative plots of all the algorithms and returns results into dictionaries for further comparison and plotting 

  '''
  scores = {'algorithm' : [], 'metric' : [], 'value' : [], 'dataset' : []} 
  
  scores_ = {'sensitivity' : [], 'specificity' : [], 'accuracy' : [], 'algorithm' : [], 'dataset' : [] }

  metric_names = ['accuracy', 'precision', 'sensitivity', 'specificity']
  ### KNN ###
  
  acc, prec, rec, sens, spec = KNN(X_train, Y_train, X_val, Y_val, X_test, Y_test, dataset)
  metrics = [acc, prec, sens, spec]

  for i, m in enumerate(metrics): 
    scores['algorithm'].append('KNN')
    scores['metric'].append(metric_names[i])
    scores['value'].append(metrics[i])
    scores['dataset'].append(dataset)
    
  scores_['sensitivity'].append(sens)
  scores_['specificity'].append(spec)
  scores_['accuracy'].append(spec)
  scores_['dataset'].append(dataset)
  scores_['algorithm'].append('KNN')
  
  
  ### SVM ###
  acc, prec, rec,sens, spec = SVM(X_train, Y_train, X_val, Y_val, X_test, Y_test, dataset)
  metrics = [acc, prec, sens, spec]

  for i, m in enumerate(metrics): 
    scores['algorithm'].append('SVM')
    scores['metric'].append(metric_names[i])
    scores['value'].append(metrics[i])
    scores['dataset'].append(dataset)
    
  scores_['sensitivity'].append(sens)
  scores_['specificity'].append(spec)
  scores_['accuracy'].append(spec)
  scores_['dataset'].append(dataset)
  scores_['algorithm'].append('SVM')
  
  ### Decision Tree ###

  acc, prec, rec, sens, spec = Decision_Tree(X_train, Y_train, X_val, Y_val, X_test, Y_test, dataset)
  metrics = [acc, prec, sens, spec]

  for i, m in enumerate(metrics): 
    scores['algorithm'].append('Decision Trees')
    scores['metric'].append(metric_names[i])
    scores['value'].append(metrics[i])
    scores['dataset'].append(dataset)

  scores_['sensitivity'].append(sens)
  scores_['specificity'].append(spec)
  scores_['accuracy'].append(spec)
  scores_['dataset'].append(dataset)
  scores_['algorithm'].append('Decision Tree')

  ### Logistic Regression ###
  acc, prec, rec, sens, spec = Logistic_Regression(X_train, Y_train, X_val, Y_val, X_test, Y_test, dataset)
  metrics = [acc, prec, sens, spec]

  for i, m in enumerate(metric_names): 
    scores['algorithm'].append('Logistic Regression')
    scores['metric'].append(metric_names[i])
    scores['value'].append(metrics[i])
    scores['dataset'].append(dataset)

  scores_['sensitivity'].append(sens)
  scores_['specificity'].append(spec)
  scores_['accuracy'].append(spec)
  scores_['dataset'].append(dataset)
  scores_['algorithm'].append('Logistic Regression')

  ### AdaBoost ###
  
  acc, prec, rec, sens, spec = AdaBoost(X_train, Y_train, X_val, Y_val, X_test, Y_test, dataset)
  metrics = [acc, prec, sens, spec]

  for i, m in enumerate(metrics): 
    scores['algorithm'].append('AdaBoost')
    scores['metric'].append(metric_names[i])
    scores['value'].append(metrics[i])
    scores['dataset'].append(dataset)

  scores_['sensitivity'].append(sens)
  scores_['specificity'].append(spec)
  scores_['accuracy'].append(spec)
  scores_['dataset'].append(dataset)
  scores_['algorithm'].append('AdaBoost')
  
  ### MLP ### 
  acc, prec, rec, sens, spec = MLP(X_train, Y_train, X_val, Y_val, X_test, Y_test, dataset)
  metrics = [acc, prec, sens, spec]

  for i, m in enumerate(metrics): 
    scores['algorithm'].append('MLP')
    scores['metric'].append(metric_names[i])
    scores['value'].append(metrics[i])
    scores['dataset'].append(dataset)

  scores_['sensitivity'].append(sens)
  scores_['specificity'].append(spec)
  scores_['accuracy'].append(spec)
  scores_['dataset'].append(dataset)
  scores_['algorithm'].append('MLP')


  return scores, scores_


if __name__ == "__main__": 


  # load npys

  X1 = np.load('../data/features_1.npy')
  Y1 = np.load('../data/target_1.npy') 

  # Split Train-Test 
  print('Dataset 1 \n')
  (X_train1, X_val1, Y_train1, Y_val1) = train_test_split(X1, Y1, test_size=0.05, random_state=42)
  (X_val1, X_test1, Y_val1, Y_test1) = train_test_split(X_val1, Y_val1, test_size=0.2, random_state=42)

  # show the sizes of each data split
  print("training data points: {}".format(len(Y_train1)))
  print("validation data points: {}".format(len(Y_val1)))
  print("testing data points: {}".format(len(Y_test1)))
  print() 


  ### RUN ML ALGORITHMS FOR ALL SETS #### 
  scores_1, scores1_ = apply_ML(X_train=X_train1, Y_train=Y_train1, X_val=X_val1, Y_val=Y_val1, X_test=X_test1, Y_test=Y_test1, dataset='dataset1')

  with open('scores_dataset1.pickle', 'wb') as handle:
      pickle.dump(scores_1, handle, protocol=pickle.HIGHEST_PROTOCOL)
  with open('sep_scores_dataset1.pickle', 'wb') as handle:
      pickle.dump(scores_1, handle, protocol=pickle.HIGHEST_PROTOCOL)

