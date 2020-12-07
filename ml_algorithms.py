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

def KNN(data, targets, dataset): 
  print('APPLY KNN ...')
  
  skf = StratifiedKFold(n_splits=4)
  # Only take the first fold.
  train_index, test_index = next(iter(skf.split(data,targets)))

  X_train = data[train_index]
  Y_train = targets[train_index]
  X_test = data[test_index]
  Y_test = targets[test_index]

  print('Test', X_test.shape, Y_test.shape)

  skf = StratifiedKFold(n_splits=2)
  val_index, _ = next(iter(skf.split(X_test,Y_test)))

  X_val = X_test[val_index]
  Y_val = Y_test[val_index]
  
  print('X_val', X_val.shape, Y_val.shape)

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
  acc, prec, rec, sens, spec = evaluate(model, X_test, Y_test, np.array(predictions), np.array(scores[:,1]), np.array(Y_test), 'knn'+dataset) 
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

def SVM(data, targets, dataset):
  print('APPLY SVM ...')
  skf = StratifiedKFold(n_splits=4)
  # Only take the first fold.
  train_index, test_index = next(iter(skf.split(data,targets)))

  X_train = data[train_index]
  Y_train = targets[train_index]
  X_test = data[test_index]
  Y_test = targets[test_index]

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


  acc, prec, rec, sens, spec = evaluate(final_model, X_test, Y_test, np.array(predictions), np.array(scores), np.array(Y_test), 'svm' +dataset) 
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

def Decision_Tree(data, targets, dataset):
  print('APPLY DECISION TREE CLASSIFIER ...')
  skf = StratifiedKFold(n_splits=4)
  # Only take the first fold.
  train_index, test_index = next(iter(skf.split(data,targets)))

  X_train = data[train_index]
  Y_train = targets[train_index]
  X_test = data[test_index]
  Y_test = targets[test_index]

  params = {'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4]}
  grid_search_cv = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1, verbose=1, cv=3)

  grid_search_cv.fit(X_train, Y_train)

  grid_search_cv.best_estimator_

  y_pred = grid_search_cv.predict(X_test)
  accuracy_score(Y_test, y_pred)

  predictions = grid_search_cv.predict(X_test)
  scores = grid_search_cv.predict_proba(X_test)
  acc, prec, rec, sens,spec = evaluate(grid_search_cv, X_test, Y_test, np.array(predictions), np.array(scores[:,1]), np.array(Y_test), 'decision_tree'+dataset) 
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

def Logistic_Regression(data, targets, dataset): 
  print('APPLY LOGISTIC REGRESSION ...')

  skf = StratifiedKFold(n_splits=4)
  # Only take the first fold.
  train_index, test_index = next(iter(skf.split(data,targets)))

  X_train = data[train_index]
  Y_train = targets[train_index]
  X_test = data[test_index]
  Y_test = targets[test_index]

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
    acc, prec, rec, sens, spec = evaluate(clf, X_test,Y_test,np.array(predictions), np.array(scores[:,1]), np.array(Y_test), 'logistic'+penalty + dataset) 
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
    plt.savefig('../plots/'+dataset +'_logisticregression_confmatrix.png')


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

def AdaBoost(data, targets, dataset): 
  print('APPLY ADABOOST ...')

  skf = StratifiedKFold(n_splits=4)
  # Only take the first fold.
  train_index, test_index = next(iter(skf.split(data,targets)))

  X_train = data[train_index]
  Y_train = targets[train_index]
  X_test = data[test_index]
  Y_test = targets[test_index]

  print('Test', X_test.shape, Y_test.shape)

  skf = StratifiedKFold(n_splits=2)
  val_index, _ = next(iter(skf.split(X_test,Y_test)))

  X_val = X_test[val_index]
  Y_val = Y_test[val_index]
  
  print('X_val', X_val.shape, Y_val.shape)

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

  plt.plot(depths, train_accuracy, '-')
  plt.plot(depths, val_accuracy, '--')
  plt.title('The accuracy of different max_depth')
  plt.savefig('../plots/'+dataset + '_accuracy_depth_adaboostmaxindex_specs.png')
  # plt.show()

  best_model = models[np.argmax(val_accuracy)]
  best_model.score(X_test, Y_test)

  predictions = best_model.predict(X_test)
  scores = best_model.predict_proba(X_test)
  acc, prec, rec, sens,spec = evaluate(best_model, X_test, Y_test, np.array(predictions), np.array(scores[:,1]), np.array(Y_test), 'adaboost' + dataset) 
  print('Test Accuracy, Precision, Recall', acc, prec, rec)
  print()

  print("Classification report for classifier %s:\n%s\n"
      % (best_model, metrics.classification_report(Y_test, predictions)))
  plt.figure() 
  disp = metrics.plot_confusion_matrix(best_model, X_test, Y_test)
  disp.figure_.suptitle("Confusion Matrix")
  print("Confusion matrix:\n%s" % disp.confusion_matrix)
  plt.savefig('../plots/'+dataset +'_adaboost_confmatrix.png')

  return acc, prec, rec, sens, spec

def MLP(data, targets, dataset): 
  print('APPLY MULTILAYER PERCEPTRON ...')

  skf = StratifiedKFold(n_splits=4)
  # Only take the first fold.
  train_index, test_index = next(iter(skf.split(data,targets)))

  X_train = data[train_index]
  Y_train = targets[train_index]
  X_test = data[test_index]
  Y_test = targets[test_index]

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

  acc, prec, rec, sens, spec = evaluate(clf, X_test, Y_test, np.array(predictions), np.array(scores[:,1]), np.array(Y_test), 'mlp' + dataset) 
  print('Test Accuracy, Precision, Recall', acc, prec, rec)
  print()


  print("Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(Y_test, predictions)))
  plt.figure() 
  disp = metrics.plot_confusion_matrix(clf, X_test, Y_test)
  disp.figure_.suptitle("Confusion Matrix")
  print("Confusion matrix:\n%s" % disp.confusion_matrix)
  plt.savefig('../plots/'+dataset +'_mlp_confmatrix.png')


  return acc, prec, rec, sens, spec 


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

def GMM_Sklearn(data, targets):
  """
  ===============
  GMM covariances
  ===============

  Demonstration of several covariances types for Gaussian mixture models.

  See :ref:`gmm` for more information on the estimator.

  Although GMM are often used for clustering, we can compare the obtained
  clusters with the actual classes from the dataset. We initialize the means
  of the Gaussians with the means of the classes from the training set to make
  this comparison valid.

  We plot predicted labels on both training and held out test data using a
  variety of GMM covariance types on the iris dataset.
  We compare GMMs with spherical, diagonal, full, and tied covariance
  matrices in increasing order of performance. Although one would
  expect full covariance to perform best in general, it is prone to
  overfitting on small datasets and does not generalize well to held out
  test data.

  On the plots, train data is shown as dots, while test data is shown as
  crosses. The iris dataset is four-dimensional. Only the first two
  dimensions are shown here, and thus some points are separated in other
  dimensions.
  """
  colors = ['navy', 'turquoise']
  target_names = ['CN', 'Dementia']
  # Break up the dataset into non-overlapping training (75%) and testing
  # (25%) sets.
  # iris = datasets.load_iris()
  skf = StratifiedKFold(n_splits=4)
  # Only take the first fold.
  train_index, test_index = next(iter(skf.split(data,targets)))

  X_train = data[train_index]
  Y_train = targets[train_index]
  X_test = data[test_index]
  Y_test = targets[test_index]

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
        print(data.shape)
        dataf = data[(targets == n)]
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


  # plt.show()

def KMEANS(data, targets, dataset, n_classes): 

  # PCA - dims 

  skf = StratifiedKFold(n_splits=4)
  # Only take the first fold.
  train_index, test_index = next(iter(skf.split(data,targets)))

  X_train = data[train_index]
  Y_train = targets[train_index]
  X_test = data[test_index]
  Y_test = targets[test_index]

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

    # cluster_centers_2d = pca.fit_transform(cluster_centers)

    # sns.scatterplot(x=cluster_centers_2d[:,0], y=cluster_centers_2d[:,1])
    # plt.show() 
    # sys.exit(0)

    prediction = kmeans.predict(X_test)
    print('Prediction', prediction.shape)


    acc = accuracy_score(y_true=Y_test, y_pred=prediction) 
    prec = sklearn.metrics.precision_score(y_true=Y_test, y_pred=prediction )
    rec = sklearn.metrics.recall_score(y_true=Y_test, y_pred=prediction)
    class_report = sklearn.metrics.classification_report(y_true=Y_test, y_pred=prediction, output_dict=True)

    sens = class_report['1']['recall']
    spec = class_report['0']['recall']

    # acc, prec, rec, sens, spec = evaluate(kmeans, X_test, Y_test, np.array(prediction), np.array(scores[:,1]), np.array(Y_test), 'kmeans' + dataset) 
    print('Test Accuracy, Precision, Recall', acc, prec, rec)
    print()

    accuracies.append(acc)
    precisions.append(prec)
    recalls.append(rec)
    senss.append(sens)
    specs.append(spec)

    # print("Classification report for classifier %s:\n%s\n"
    #     % (kmeans, metrics.classification_report(Y_test, prediction)))
    # plt.figure() 
    # disp = metrics.plot_confusion_matrix(kmeans, X_test, Y_test)
    # disp.figure_.suptitle("Confusion Matrix")
    # print("Confusion matrix:\n%s" % disp.confusion_matrix)
    # plt.savefig('../plots/'+dataset +'_kmeans_confmatrix'+str(pcadim)+'.png')

  plt.figure() 
  sns.lineplot(x=pca_dims, y=accuracies) 
  plt.xlabel('pca dimensions')
  plt.ylabel('accuracy')
  # plt.show()
  plt.savefig('../plots/'+dataset +'_kmeans_accuracy.png')

  plt.figure() 
  sns.lineplot(x=pca_dims, y=senss) 
  sns.lineplot(x=pca_dims, y=specs) 
  plt.xlabel('pca dimensions')
  plt.ylabel('sensitivity')
  # plt.show()
  plt.savefig('../plots/'+dataset +'_kmeans_sensitivity_specificity.png')
    
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

def LMM(): 
  pass 

print('RUN SUCCESSFULLY!')


def apply_ML(data, targets, dataset): 
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
  scores_['accuracy'].append(spec)
  scores_['dataset'].append(dataset)
  scores_['algorithm'].append('KMEANS')


  ### GMM ###
  GMM_Sklearn(data, targets)
  
  ### KNN ###
  acc, prec, rec, sens, spec = KNN(data, targets, dataset)
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
  acc, prec, rec,sens, spec = SVM(data, targets, dataset)
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
  acc, prec, rec, sens, spec = Decision_Tree(data, targets, dataset)
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
  acc, prec, rec, sens, spec = Logistic_Regression(data, targets, dataset)
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
  acc, prec, rec, sens, spec = AdaBoost(data, targets, dataset)
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
  acc, prec, rec, sens, spec = MLP(data, targets, dataset)
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

  X1 = np.load('../data/features_1.npy')
  Y1 = np.load('../data/target_1.npy') 

  scores_1, scores1_ = apply_ML(data=X1, targets=Y1, dataset='dataset1')

  with open('scores_dataset1.pickle', 'wb') as handle:
      pickle.dump(scores_1, handle, protocol=pickle.HIGHEST_PROTOCOL)
  with open('sep_scores_dataset1.pickle', 'wb') as handle:
      pickle.dump(scores_1, handle, protocol=pickle.HIGHEST_PROTOCOL)

  # PLOT SENSITIVITY, SPECIFICITY FOR ALL ALGOS
  plt.figure()
  data = pd.DataFrame(data=scores_1)
  sns.scatterplot("algorithm", "value", hue='metric', data=scores_1)
  plt.title('Evaluation Metrics for ML Algorithms')
  plt.savefig("../plots/all_algo_metrics_dataset1.png")
  # plt.show()

  print('Dataset 2 \n')
  X2 = np.load('../data/mean_imputation_features_2.npy')
  Y2 = np.load('../data/target_1.npy') 
  print('Features', X2.shape)
  print('Targets', Y2.shape)

  scores_2, scores2_ = apply_ML(data=X2, targets=Y1, dataset='dataset2')

  with open('scores_dataset2.pickle', 'wb') as handle:
      pickle.dump(scores_2, handle, protocol=pickle.HIGHEST_PROTOCOL)
  with open('sep_scores_dataset2.pickle', 'wb') as handle:
      pickle.dump(scores_2, handle, protocol=pickle.HIGHEST_PROTOCOL)
  
  # PLOT SENSITIVITY, SPECIFICITY FOR ALL ALGOS
  plt.figure()
  data = pd.DataFrame(data=scores_2)
  sns.scatterplot("algorithm", "value", hue='metric', data=scores_2)
  plt.title('Evaluation Metrics for ML Algorithms')
  plt.savefig("../plots/all_algo_metrics_dataset2.png")
  # plt.show()


  print('Dataset 3 \n')
  X3 = np.load('../data/regression_imputation_features_2.npy')
  Y3 = np.load('../data/target_1.npy') 
  print('Features', X3.shape)
  print('Targets', Y3.shape)

  scores_3, scores3_ = apply_ML(data=X3, targets=Y3, dataset='dataset3')

  with open('scores_dataset3.pickle', 'wb') as handle:
      pickle.dump(scores_3, handle, protocol=pickle.HIGHEST_PROTOCOL)
  with open('sep_scores_dataset3.pickle', 'wb') as handle:
      pickle.dump(scores_3, handle, protocol=pickle.HIGHEST_PROTOCOL)
  

  # PLOT SENSITIVITY, SPECIFICITY FOR ALL ALGOS
  plt.figure()
  data = pd.DataFrame(data=scores_2)
  sns.scatterplot("algorithm", "value", hue='metric', data=scores_3)
  plt.title('Evaluation Metrics for ML Algorithms')
  plt.savefig("../plots/all_algo_metrics_dataset3.png")
  # plt.show()


  
