import numpy as np
import pandas as pd
import sklearn 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle 
import sys 
import seaborn 
import matplotlib as mpl 
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
from sklearn.decomposition import PCA
from time import time
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold

plt.rcParams["axes.grid"] = False

def KNN(data, targets, dataset, n_classes): 
  print('APPLY KNN ...')
  X_train, Y_train = data[0], targets[0]
  X_val, Y_val = data[1], targets[1]
  X_test, Y_test = data[2], targets[2]

  k_values = range(1, 30, 2)
  f1scores = [] 
  for k in range(1, 30, 2):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, Y_train)
    prediction = model.predict(X_val)

    score = sklearn.metrics.f1_score(Y_val, prediction,  average='weighted') 

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
  acc, prec, rec, sens, spec = evaluate(model, X_test, Y_test, np.array(predictions), np.array(scores[:,1]), np.array(Y_test), n_classes, 'knn'+dataset) 
  print('Test Accuracy, Precision, Recall', acc, prec, rec)
  print()

  print("Classification report for classifier %s:\n%s\n"
      % (model, metrics.classification_report(Y_test, predictions)))
  
  plt.figure()
  disp = metrics.plot_confusion_matrix(model, X_test, Y_test)
  disp.figure_.suptitle("Confusion Matrix")
  print("Confusion matrix:\n%s" % disp.confusion_matrix)
  plt.savefig('../plots/'+dataset +'_knn_confmatrix.png')

  return acc, prec, rec, sens, spec 

def SVM(data, targets, dataset, n_classes):
  print('APPLY SVM ...')
  X_train, Y_train = data[0], targets[0]
  X_val, Y_val = data[1], targets[1]
  X_test, Y_test = data[2], targets[2]

  ### Grid Search to some parameters ####
  # params_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
  #                    'C': [1, 10, 100, 1000]},
  #                   {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

  final_model = make_pipeline(StandardScaler(), SVC(gamma='scale'))

  # svm_model = GridSearchCV(SVC(), params_grid, cv=5)
  final_model.fit(X_train, Y_train)

  # # View the accuracy score
  # print('Best score for training data:', svm_model.best_score_,"\n") 

  # # View the best parameters for the model found using grid search
  # print('Best C:',svm_model.best_estimator_.C,"\n") 
  # print('Best Kernel:',svm_model.best_estimator_.kernel,"\n")
  # print('Best Gamma:',svm_model.best_estimator_.gamma,"\n")

  # final_model = svm_model.best_estimator_

  predictions = final_model.predict(X_test)
  scores = final_model.decision_function(X_test)

  # evaluate the uncertainty 
  # plot_uncertainty(groundtruth=Y_test, uncertainty=scores, path='svm_' + dataset)


  acc, prec, rec, sens, spec = evaluate(final_model, X_test, Y_test, np.array(predictions), np.array(scores), np.array(Y_test), n_classes, 'svm' +dataset) 
  print('Test Accuracy, Precision, Recall', acc, prec, rec)
  print()


  print("Classification report for classifier %s:\n%s\n"
      % (final_model, metrics.classification_report(Y_test, predictions)))
  plt.figure() 
  disp = metrics.plot_confusion_matrix(final_model, X_test, Y_test)
  disp.figure_.suptitle("Confusion Matrix")
  print("Confusion matrix:\n%s" % disp.confusion_matrix)
  plt.savefig('../plots/'+dataset +'_svm_confmatrix.png')


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

def Decision_Tree(data, targets, dataset, n_classes):
  print('APPLY DECISION TREE CLASSIFIER ...')
  X_train, Y_train = data[0], targets[0]
  X_val, Y_val = data[1], targets[1]
  X_test, Y_test = data[2], targets[2]

  params = {'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4]}
  grid_search_cv = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1, verbose=1, cv=3)

  grid_search_cv.fit(X_train, Y_train)

  grid_search_cv.best_estimator_

  y_pred = grid_search_cv.predict(X_test)
  accuracy_score(Y_test, y_pred)

  predictions = grid_search_cv.predict(X_test)
  scores = grid_search_cv.predict_proba(X_test)
  acc, prec, rec, sens,spec = evaluate(grid_search_cv, X_test, Y_test, np.array(predictions), np.array(scores[:,1]), np.array(Y_test), n_classes, 'decision_tree'+dataset) 
  print('Test Accuracy, Precision, Recall', acc, prec, rec)
  print()


  print("Classification report for classifier %s:\n%s\n"
      % (grid_search_cv, metrics.classification_report(Y_test, predictions)))
  plt.figure() 
  disp = metrics.plot_confusion_matrix(grid_search_cv, X_test, Y_test)
  disp.figure_.suptitle("Confusion Matrix")
  print("Confusion matrix:\n%s" % disp.confusion_matrix)
  plt.savefig('../plots/'+dataset +'_decisiontree_confmatrix.png')

  return acc, prec, rec, sens, spec

def Logistic_Regression(data, targets, dataset, n_classes): 
  print('APPLY LOGISTIC REGRESSION ...')
  X_train, Y_train = data[0], targets[0]
  X_val, Y_val = data[1], targets[1]
  X_test, Y_test = data[2], targets[2]

  penalties = ['l2', 'l1']

  accuracies = [] 
  precisions = [] 
  senss, specs, recalls = [],[], [] 

  for penalty in penalties : 

    clf = make_pipeline(StandardScaler(),LogisticRegression(random_state=0, penalty=penalty, solver='liblinear'))

    logisticregression_score = cross_validation_roc_auc(classifier=clf, X_train=X_train, Y_train=Y_train, algo='logistic_regression')

    # scores in cross-validation 
    scores = cross_val_score(clf, X_train, Y_train, cv=50)

    # mean cross val score
    print('Mean Cross-Val score', np.mean(scores))
    data = {'cross': range(50), 'cv_scores' : scores} 
    sns.lineplot(x='cross', y='cv_scores', data=data)
    plt.savefig('../plots/cross_val_score_logistic_regression'+dataset+'.png')

    clf.fit(X_train, Y_train)

    predictions = clf.predict(X_test)
    scores = clf.predict_proba(X_test)
    acc, prec, rec, sens, spec = evaluate(clf, X_test,Y_test,np.array(predictions), np.array(scores[:,1]), np.array(Y_test), n_classes, 'logistic'+penalty + dataset) 
    print('Test Accuracy, Precision, Recall', acc, prec, rec)
    print()

    accuracies.append(acc)
    precisions.append(prec)
    recalls.append(rec)
    senss.append(sens)
    specs.append(spec)

    print("Classification report for classifier %s:\n%s\n"
        % (clf, metrics.classification_report(Y_test, predictions)))
    plt.figure() 
    disp = metrics.plot_confusion_matrix(clf, X_test, Y_test)
    disp.figure_.suptitle("Confusion Matrix")
    print("Confusion matrix:\n%s" % disp.confusion_matrix)
    plt.savefig('../plots/final_results/baseline_binary_class/'+dataset +'_logisticregression_confmatrix.png')


  maxindex_acc = accuracies.index(max(accuracies))
  maxindex_pr = precisions.index(max(precisions))
  maxindex_sens = senss.index(max(senss))
  maxindex_specs = specs.index(max(specs))

  print(maxindex_acc, maxindex_pr,maxindex_sens, maxindex_specs)

  acc = accuracies[0]
  prec = precisions[0]
  rec = recalls[0]
  sens = senss[0]
  spec = specs[0]

  return acc, prec, rec, sens, spec

def AdaBoost(data, targets, dataset, n_classes): 
  print('APPLY ADABOOST ...')
  X_train, Y_train = data[0], targets[0]
  X_val, Y_val = data[1], targets[1]
  X_test, Y_test = data[2], targets[2]

  # create a list of all possible depth values
  depths = [1,3,5,8]
  models = []
  train_accuracy = []
  val_accuracy = []
  # create a list of models 
  for depth in depths:
    print('Depth', depth)
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

  plt.figure() 
  plt.plot(depths, train_accuracy, '-')
  plt.plot(depths, val_accuracy, '--')
  plt.title('The accuracy of different max_depth')
  plt.savefig('../plots/final_results/baseline_binary_class'+dataset + '_accuracy_depth_adaboostmaxindex_specs.png')
  # plt.show()

  best_model = models[np.argmax(val_accuracy)]
  best_model.score(X_test, Y_test)

  predictions = best_model.predict(X_test)
  scores = best_model.predict_proba(X_test)
  acc, prec, rec, sens,spec = evaluate(best_model, X_test, Y_test, np.array(predictions), np.array(scores[:,1]), np.array(Y_test), n_classes, 'adaboost' + dataset) 
  print('Test Accuracy, Precision, Recall', acc, prec, rec)
  print()

  print("Classification report for classifier %s:\n%s\n"
      % (best_model, metrics.classification_report(Y_test, predictions)))
  plt.figure() 
  disp = metrics.plot_confusion_matrix(best_model, X_test, Y_test)
  disp.figure_.suptitle("Confusion Matrix")
  print("Confusion matrix:\n%s" % disp.confusion_matrix)
  plt.savefig('../plots/final_results/baseline_binary_class/'+dataset +'_adaboost_confmatrix.png')

  return acc, prec, rec, sens, spec

def MLP(data, targets, dataset, n_classes): 
  print('APPLY MULTILAYER PERCEPTRON ...')

  X_train, Y_train = data[0], targets[0]
  X_val, Y_val = data[1], targets[1]
  X_test, Y_test = data[2], targets[2]

  clf = make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=200, random_state=1, max_iter=300, learning_rate='adaptive'))

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

  Y_test = list(Y_test) 

  assert len(predictions) == len(scores)
  assert len(scores) == len(Y_test)

  acc, prec, rec, sens, spec = evaluate(clf, X_test, Y_test, np.array(predictions), np.array(scores[:,1]), np.array(Y_test), n_classes, 'mlp' + dataset) 
  print('Test Accuracy, Precision, Recall', acc, prec, rec)
  print()


  print("Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(Y_test, predictions)))
  plt.figure() 
  disp = metrics.plot_confusion_matrix(clf, X_test, Y_test)
  disp.figure_.suptitle("Confusion Matrix")
  print("Confusion matrix:\n%s" % disp.confusion_matrix)
  plt.savefig('../plots/final_results/baseline_binary_class/'+dataset +'_mlp_confmatrix.png')


  return acc, prec, rec, sens, spec 

def GMM_Sklearn(data, targets, colors,dataset, target_names):

  print('APPLY GMM...')
  X_train, Y_train = data[0], targets[0]
  X_val, Y_val = data[1], targets[1]
  X_test, Y_test = data[2], targets[2]

  n_classes = len(np.unique(Y_train))

  # Try GMMs using different types of covariances.
  estimators = {cov_type: GaussianMixture(n_components=n_classes,
                covariance_type=cov_type, max_iter=20, random_state=0)
                for cov_type in ['spherical', 'diag', 'tied', 'full']}

  n_estimators = len(estimators)

  plt.figure(figsize=(3 * n_estimators // 2, 6))
  plt.subplots_adjust(bottom=.01, top=0.95, hspace=.15, wspace=.05,
                      left=.01, right=.99)


  for index, (name, estimator) in enumerate(estimators.items()):
    # Since we have class labels for the training data, we can
    # initialize the GMM parameters in a supervised manner.
    estimator.means_init = np.array([X_train[Y_train == i].mean(axis=0)
                                    for i in range(n_classes)])

    # Train the other parameters using the EM algorithm.
    estimator.fit(X_train)

    h = plt.subplot(2, n_estimators // 2, index + 1)
    make_ellipses(estimator, h, colors)

    for n, color in enumerate(colors):
      dataf = X_train[(Y_train == n)]
      plt.scatter(dataf[:, 0], dataf[:, 1], s=0.8, color=color,
                  label=target_names[n])
    # Plot the test data with crosses
    for n, color in enumerate(colors):
      dataf = X_test[Y_test == n]
      plt.scatter(dataf[:, 0], dataf[:, 1], marker='x', color=color)

    y_train_pred = estimator.predict(X_train)
    train_accuracy = np.mean(y_train_pred.ravel() == Y_train.ravel()) * 100
    plt.text(0.05, 0.9, 'Train accuracy: %.1f' % train_accuracy,
            transform=h.transAxes)

    y_test_pred = estimator.predict(X_test)
    test_accuracy = np.mean(y_test_pred.ravel() == Y_test.ravel()) * 100
    plt.text(0.05, 0.8, 'Test accuracy: %.1f' % test_accuracy,
            transform=h.transAxes)

    plt.xticks(())
    plt.yticks(())
    plt.title(name)

  plt.legend(scatterpoints=1, loc='lower right', prop=dict(size=12))

  ### APPLY GMM ### 
  estimator = GaussianMixture(n_components=n_classes,
                covariance_type='tied', max_iter=20, random_state=0)

  estimator.means_init = np.array([X_train[Y_train == i].mean(axis=0)
                                      for i in range(n_classes)])

  # Train the other parameters using the EM algorithm.
  estimator.fit(X_train)
  predictions = estimator.predict(X_test)
  scores = estimator.predict_proba(X_test)

  Y_test = list(Y_test) 

  assert len(predictions) == len(scores)
  assert len(scores) == len(Y_test)

  acc, prec, rec, sens, spec = evaluate(estimator, X_test, Y_test, np.array(predictions), np.array(scores[:,1]), np.array(Y_test), n_classes, 'gmm' + dataset) 
  print('Test Accuracy, Precision, Recall', acc, prec, rec)
  print()


  # print("Classification report for classifier %s:\n%s\n"
  #     % (estimator, metrics.classification_report(Y_test, predictions)))
  # plt.figure() 
  # disp = metrics.plot_confusion_matrix(estimator, X_test, Y_test)
  # disp.figure_.suptitle("Confusion Matrix")
  # print("Confusion matrix:\n%s" % disp.confusion_matrix)
  # plt.savefig('../plots/'+dataset +'_Gmm_confmatrix.png')

  return acc, prec, rec, sens, spec 

def PCAKMEANS(data, targets, dataset, n_classes): 
  print('APPLY KMEANS...')
  X_train, Y_train = data[0], targets[0]
  X_val, Y_val = data[1], targets[1]
  X_test, Y_test = data[2], targets[2]
  # PCA - dims

  pca_dims = [2,10, 20, 50]

  accuracies = [] 
  precisions = [] 
  senss, specs, recalls = [],[], [] 

  for pcadim in pca_dims: 

    pca = PCA(n_components=pcadim).fit(X_train)

    kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X_train)
    kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    print('Cluster Centers', cluster_centers.shape)

    prediction = kmeans.predict(X_test)
    print('Prediction', prediction.shape)

    acc = accuracy_score(y_true=Y_test, y_pred=prediction) 
    prec = sklearn.metrics.precision_score(y_true=Y_test, y_pred=prediction,  average='weighted' )
    rec = sklearn.metrics.recall_score(y_true=Y_test, y_pred=prediction,  average='weighted')
    class_report = sklearn.metrics.classification_report(y_true=Y_test, y_pred=prediction, output_dict=True)

    sens = class_report['1']['recall']
    spec = class_report['0']['recall']

    print('Test Accuracy, Precision, Recall', acc, prec, rec)
    print()

    accuracies.append(acc)
    precisions.append(prec)
    recalls.append(rec)
    senss.append(sens)
    specs.append(spec)

  plt.figure() 
  sns.lineplot(x=pca_dims, y=accuracies) 
  plt.xlabel('pca dimensions')
  plt.ylabel('accuracy')
  # plt.show()
  # plt.savefig('../plots/'+dataset +'_kmeans_accuracy.png')

  plt.figure() 
  sns.lineplot(x=pca_dims, y=senss) 
  sns.lineplot(x=pca_dims, y=specs) 
  plt.xlabel('pca dimensions')
  plt.ylabel('sensitivity')
  # plt.show()
  # plt.savefig('../plots/'+dataset +'_kmeans_sensitivity_specificity.png')
    
  maxindex_acc = accuracies.index(max(accuracies))
  maxindex_pr = precisions.index(max(precisions))
  maxindex_sens = senss.index(max(senss))
  maxindex_specs = specs.index(max(specs))

  print(maxindex_acc, maxindex_pr,maxindex_sens, maxindex_specs)

  acc = accuracies[0]
  prec = precisions[0]
  rec = recalls[0]
  sens = senss[0]
  spec = specs[0]

  return acc, prec, rec, sens, spec

def KMEANS(data, targets, dataset, n_classes): 
  print('APPLY KMEANS...')
  X_train, Y_train = data[0], targets[0]
  X_val, Y_val = data[1], targets[1]
  X_test, Y_test = data[2], targets[2]

  kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X_train)
  kmeans.labels_
  cluster_centers = kmeans.cluster_centers_
  print('Cluster Centers', cluster_centers.shape)

  prediction = kmeans.predict(X_test)
  print('Prediction', prediction.shape)
  scores = kmeans.score(X_test)

  acc = accuracy_score(y_true=Y_test, y_pred=prediction) 
  prec = sklearn.metrics.precision_score(y_true=Y_test, y_pred=prediction,  average='weighted' )
  rec = sklearn.metrics.recall_score(y_true=Y_test, y_pred=prediction,  average='weighted')
  class_report = sklearn.metrics.classification_report(y_true=Y_test, y_pred=prediction, output_dict=True)

  sens = class_report['1']['recall']
  spec = class_report['0']['recall']

  print('Test Accuracy, Precision, Recall', acc, prec, rec)
  print()

  return acc, prec, rec, sens, spec


def apply_ML(data, targets, dataset, colors, target_names, n_classes): 
  ''' 
  X_train, Y_train, X_val, Y_val, X_test, Y_test : sets 
  produces evaluation plots, cumulative plots of all the algorithms and returns results into dictionaries for further comparison and plotting 

  '''
  scores = {'algorithm' : [], 'metric' : [], 'value' : [], 'dataset' : []} 
  
  scores_ = {'sensitivity' : [], 'specificity' : [], 'accuracy' : [], 'algorithm' : [], 'dataset' : [] }

  metric_names = ['accuracy', 'precision', 'sensitivity', 'specificity']
  
  ### K-MEANS ### 
  acc, prec, rec, sens, spec = KMEANS(data, targets, dataset,2)
  metrics = [acc, prec, sens, spec]

  for i, m in enumerate(metrics): 
    scores['algorithm'].append('KMEANS')
    scores['metric'].append(metric_names[i])
    scores['value'].append(metrics[i])
    scores['dataset'].append(dataset)
    
  scores_['sensitivity'].append(sens)
  scores_['specificity'].append(spec)
  scores_['accuracy'].append(acc)
  scores_['dataset'].append(dataset)
  scores_['algorithm'].append('KMEANS')

  ### GMM ###
  acc, prec, rec, sens, spec = GMM_Sklearn(data, targets, colors, dataset, target_names)
  metrics = [acc, prec, sens, spec]

  for i, m in enumerate(metrics): 
    scores['algorithm'].append('GMM')
    scores['metric'].append(metric_names[i])
    scores['value'].append(metrics[i])
    scores['dataset'].append(dataset)
    
  scores_['sensitivity'].append(sens)
  scores_['specificity'].append(spec)
  scores_['accuracy'].append(acc)
  scores_['dataset'].append(dataset)
  scores_['algorithm'].append('GMM')
  
  ### KNN ###
  acc, prec, rec, sens, spec = KNN(data, targets, dataset, n_classes) 
  metrics = [acc, prec, sens, spec]

  for i, m in enumerate(metrics): 
    scores['algorithm'].append('KNN')
    scores['metric'].append(metric_names[i])
    scores['value'].append(metrics[i])
    scores['dataset'].append(dataset)
    
  scores_['sensitivity'].append(sens)
  scores_['specificity'].append(spec)
  scores_['accuracy'].append(acc)
  scores_['dataset'].append(dataset)
  scores_['algorithm'].append('KNN')
  
  
  ### SVM ###
  acc, prec, rec,sens, spec = SVM(data, targets, dataset, n_classes)
  metrics = [acc, prec, sens, spec]

  for i, m in enumerate(metrics): 
    scores['algorithm'].append('SVM')
    scores['metric'].append(metric_names[i])
    scores['value'].append(metrics[i])
    scores['dataset'].append(dataset)
    
  scores_['sensitivity'].append(sens)
  scores_['specificity'].append(spec)
  scores_['accuracy'].append(acc)
  scores_['dataset'].append(dataset)
  scores_['algorithm'].append('SVM')
  
  
  ### Decision Tree ###
  acc, prec, rec, sens, spec = Decision_Tree(data, targets, dataset, n_classes)
  metrics = [acc, prec, sens, spec]

  for i, m in enumerate(metrics): 
    scores['algorithm'].append('DecTrees')
    scores['metric'].append(metric_names[i])
    scores['value'].append(metrics[i])
    scores['dataset'].append(dataset)

  scores_['sensitivity'].append(sens)
  scores_['specificity'].append(spec)
  scores_['accuracy'].append(acc)
  scores_['dataset'].append(dataset)
  scores_['algorithm'].append('DecTrees')
  
  ### Logistic Regression ###
  acc, prec, rec, sens, spec = Logistic_Regression(data, targets, dataset, n_classes)
  metrics = [acc, prec, sens, spec]

  for i, m in enumerate(metric_names): 
    scores['algorithm'].append('LogRegress')
    scores['metric'].append(metric_names[i])
    scores['value'].append(metrics[i])
    scores['dataset'].append(dataset)

  scores_['sensitivity'].append(sens)
  scores_['specificity'].append(spec)
  scores_['accuracy'].append(acc)
  scores_['dataset'].append(dataset)
  scores_['algorithm'].append('LogRegress')
  
  ### AdaBoost ###
  acc, prec, rec, sens, spec = AdaBoost(data, targets, dataset, n_classes)
  metrics = [acc, prec, sens, spec]

  for i, m in enumerate(metrics): 
    scores['algorithm'].append('AdaBoost')
    scores['metric'].append(metric_names[i])
    scores['value'].append(metrics[i])
    scores['dataset'].append(dataset)

  scores_['sensitivity'].append(sens)
  scores_['specificity'].append(spec)
  scores_['accuracy'].append(acc)
  scores_['dataset'].append(dataset)
  scores_['algorithm'].append('AdaBoost')
  
  ### MLP ### 
  acc, prec, rec, sens, spec = MLP(data, targets, dataset, n_classes)
  metrics = [acc, prec, sens, spec]

  for i, m in enumerate(metrics): 
    scores['algorithm'].append('MLP')
    scores['metric'].append(metric_names[i])
    scores['value'].append(metrics[i])
    scores['dataset'].append(dataset)

  scores_['sensitivity'].append(sens)
  scores_['specificity'].append(spec)
  scores_['accuracy'].append(acc)
  scores_['dataset'].append(dataset)
  scores_['algorithm'].append('MLP')
  
  
  return scores, scores_

if __name__ == "__main__": 

  datasets = [('../data/features_1.npy', '../data/target_1.npy'), ('../data/mean_imputation_features_2.npy', '../data/target_1.npy' )]
  datasets.append(('../data/regression_imputation_features_2.npy', '../data/target_1.npy' ))
  resultnames = [ 'dataset1', 'dataset2', 'dataset3' ]

  for i, (dataset, resultname) in enumerate(zip(datasets, resultnames)): 

    features, targets = dataset

    X = np.load(features)
    Y = np.load(targets) 

    n_classes = len(np.unique(Y))
  
    print(n_classes)

    ### Stratify DATA ### 
    skf = StratifiedKFold(n_splits=4)
    # Only take the first fold.
    train_index, test_index = next(iter(skf.split(X,Y)))

    X_train = X[train_index]
    Y_train = Y[train_index]
    X_test = X[test_index]
    Y_test = Y[test_index]

    print('CN', Y_train.tolist().count(0))
    print('Dementia', Y_train.tolist().count(1)) 
  

    print('Test', X_test.shape, Y_test.shape)

    skf = StratifiedKFold(n_splits=2)
    val_index, _ = next(iter(skf.split(X_test,Y_test)))

    X_val = X_test[val_index]
    Y_val = Y_test[val_index]

    print('X_val', X_val.shape, Y_val.shape)

    X = [X_train, X_val, X_test]
    Y = [Y_train, Y_val, Y_test]

    colors = ['navy', 'turquoise']
    target_names = ['CN', 'Dementia']
  
    scores_1, scores1_ = apply_ML(data=X, targets=Y, dataset=resultname, colors=colors, target_names=target_names, n_classes=n_classes)

    with open('scores_'+ resultname+'.pickle', 'wb') as handle:
        pickle.dump(scores_1, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('sep_scores_'+ resultname+'.pickle', 'wb') as handle:
        pickle.dump(scores1_, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('sensitivity')
    print(scores1_['algorithm'])
    print(scores1_['sensitivity'])

    continue 

    # PLOT SENSITIVITY, SPECIFICITY FOR ALL ALGOS
    plt.figure()
    data = pd.DataFrame(data=scores_1)
    sns.scatterplot("algorithm", "value", hue='metric', data=scores_1)
    plt.xticks(rotation=45)
    plt.title('Evaluation Metrics for ML Algorithms')
    plt.savefig("../plots/final_results/baseline_binary_class/all_algo_metrics_"+ resultname+".png")
    # plt.show()





  
