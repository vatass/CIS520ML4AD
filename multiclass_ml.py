import numpy as np
import matplotlib as mpl 
import matplotlib.pyplot as plt
import seaborn as sns 
import pickle 
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from auxfunctions import *
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_val_score

def MultiKNN(data, targets, dataset, n_classes):
    print('APPLY KNN ...')
  
    X_train, Y_train = data[0], targets[0]
    X_val, Y_val = data[1], targets[1]
    X_test, Y_test = data[2], targets[2]

    model = KNeighborsClassifier(n_neighbors=20)

    clf = OneVsRestClassifier(model).fit(X_train, Y_train)

    predictions = clf.predict(X_test)
    scores = clf.predict_proba(X_test)

    print('Groundtruth', Y_test.shape)
    print('Predictions', predictions.shape)
    print('Scores', scores.shape)

    accuracy, class_report = multiclass_evaluate(clf, X_test, Y_test, np.array(predictions), np.array(scores), np.array(Y_test), n_classes, 'knn'+dataset) 

    sensitivity = [class_report['0']['recall'], class_report['1']['recall'], class_report['2']['recall']]

    confusion_matrix = sklearn.metrics.confusion_matrix(y_true=Y_test, y_pred=predictions)

    specificity0 = np.sum(confusion_matrix[1:,1:])/(np.sum(confusion_matrix[1:,1:]) + confusion_matrix[1,0] + confusion_matrix[2,0])
    Tn1 = confusion_matrix[0,0] + confusion_matrix[0,-1] + confusion_matrix[-1,0] + confusion_matrix[-1,-1]   
    specificity1 = Tn1/(Tn1 + confusion_matrix[0,1] + confusion_matrix[2,1]) 
    specificity2 = np.sum(confusion_matrix[:1,:1])/(np.sum(confusion_matrix[:1,:1]) + confusion_matrix[0,-1] + confusion_matrix[1,-1])

    specificity = [specificity0, specificity1, specificity2]
    assert specificity2 < 1 and specificity1 < 1 and specificity0 < 1 


    print("Classification report for classifier %s:\n%s\n"
        % (model, metrics.classification_report(Y_test, predictions)))

    plt.figure()
    disp = metrics.plot_confusion_matrix(clf, X_test, Y_test)
    disp.figure_.suptitle("Confusion Matrix")
    print("Confusion matrix:\n%s" % disp.confusion_matrix)
    plt.savefig('../plots/'+dataset +'_knn_confmatrix.png')
    
    return accuracy, sensitivity, specificity
    
    pass 

def MultiSVM(data, targets, dataset, n_classes): 
    print('APPLY SVM ...')
    X_train, Y_train = data[0], targets[0]
    X_val, Y_val = data[1], targets[1]
    X_test, Y_test = data[2], targets[2]

    model = make_pipeline(StandardScaler(), SVC(gamma='scale'))

    clf = OneVsRestClassifier(model).fit(X_train, Y_train) 

    predictions = clf.predict(X_test)
    scores = clf.decision_function(X_test)

    accuracy, class_report = multiclass_evaluate(clf, X_test, Y_test, np.array(predictions), np.array(scores), np.array(Y_test), n_classes, 'svm'+dataset) 

    sensitivity = [class_report['0']['recall'], class_report['1']['recall'], class_report['2']['recall']]

    confusion_matrix = sklearn.metrics.confusion_matrix(y_true=Y_test, y_pred=predictions)

    print('Confusion Matrix', confusion_matrix)
    
    specificity0 = np.sum(confusion_matrix[1:,1:])/(np.sum(confusion_matrix[1:,1:]) + confusion_matrix[1,0] + confusion_matrix[2,0])
    Tn1 = confusion_matrix[0,0] + confusion_matrix[0,-1] + confusion_matrix[-1,0] + confusion_matrix[-1,-1]   
    specificity1 = Tn1/(Tn1 + confusion_matrix[0,1] + confusion_matrix[2,1]) 
    specificity2 = np.sum(confusion_matrix[:1,:1])/(np.sum(confusion_matrix[:1,:1]) + confusion_matrix[0,-1] + confusion_matrix[1,-1])

    specificity = [specificity0, specificity1, specificity2]
    assert specificity2 < 1 and specificity1 < 1 and specificity0 < 1 


    print("Classification report for classifier %s:\n%s\n"
        % (model, metrics.classification_report(Y_test, predictions)))

    plt.figure()
    disp = metrics.plot_confusion_matrix(clf, X_test, Y_test)
    disp.figure_.suptitle("Confusion Matrix")
    print("Confusion matrix:\n%s" % disp.confusion_matrix)
    plt.savefig('../plots/'+dataset +'_svm_confmatrix.png')
    
    return accuracy, sensitivity, specificity

def MultiDecision_Tree(data, targets, dataset,  n_classes): 

    print('APPLY DECISION TREE CLASSIFIER ...')
    X_train, Y_train = data[0], targets[0]
    X_val, Y_val = data[1], targets[1]
    X_test, Y_test = data[2], targets[2]

    params = {'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4]}
    grid_search_cv = OneVsRestClassifier(GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1, verbose=1, cv=3))

    grid_search_cv.fit(X_train, Y_train)

    grid_search_cv.best_estimator_

    y_pred = grid_search_cv.predict(X_test)
    accuracy_score(Y_test, y_pred)

    predictions = grid_search_cv.predict(X_test)
    scores = grid_search_cv.predict_proba(X_test)

    accuracy, class_report = multiclass_evaluate(clf, X_test, Y_test, np.array(predictions), np.array(scores), np.array(Y_test), n_classes, 'knn'+dataset) 

    sensitivity = [class_report['0']['recall'], class_report['1']['recall'], class_report['2']['recall']]

    confusion_matrix = sklearn.metrics.confusion_matrix(y_true=Y_test, y_pred=predictions)

    print('Confusion Matrix', confusion_matrix)
    
    specificity0 = np.sum(confusion_matrix[1:,1:])/(np.sum(confusion_matrix[1:,1:]) + confusion_matrix[1,0] + confusion_matrix[2,0])
    Tn1 = confusion_matrix[0,0] + confusion_matrix[0,-1] + confusion_matrix[-1,0] + confusion_matrix[-1,-1]   
    specificity1 = Tn1/(Tn1 + confusion_matrix[0,1] + confusion_matrix[2,1]) 
    specificity2 = np.sum(confusion_matrix[:1,:1])/(np.sum(confusion_matrix[:1,:1]) + confusion_matrix[0,-1] + confusion_matrix[1,-1])

    specificity = [specificity0, specificity1, specificity2]

    assert specificity2 < 1 and specificity1 < 1 and specificity0 < 1 


    print("Classification report for classifier %s:\n%s\n"
        % (model, metrics.classification_report(Y_test, predictions)))

    plt.figure()
    disp = metrics.plot_confusion_matrix(clf, X_test, Y_test)
    disp.figure_.suptitle("Confusion Matrix")
    print("Confusion matrix:\n%s" % disp.confusion_matrix)
    plt.savefig('../plots/'+dataset +'_svm_confmatrix.png')
    
    return accuracy, sensitivity, specificity
    
def MultiMLP(data, targets, dataset, n_classes):
    print('APPLY MULTILAYER PERCEPTRON ...')

    X_train, Y_train = data[0], targets[0]
    X_val, Y_val = data[1], targets[1]
    X_test, Y_test = data[2], targets[2]

    clf = make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=200, random_state=1, max_iter=300, learning_rate='adaptive'))
    clf.fit(X_train, Y_train)
   
    predictions = clf.predict(X_test)
    scores = clf.predict_proba(X_test)

    Y_test = list(Y_test) 

    assert len(predictions) == len(scores)
    assert len(scores) == len(Y_test)

    accuracy, class_report = multiclass_evaluate(clf, X_test, Y_test, np.array(predictions), np.array(scores), np.array(Y_test), n_classes, 'mlp'+dataset) 

    sensitivity = [class_report['0']['recall'], class_report['1']['recall'], class_report['2']['recall']]

    confusion_matrix = sklearn.metrics.confusion_matrix(y_true=Y_test, y_pred=predictions)

    print('Confusion Matrix', confusion_matrix)
    
    specificity0 = np.sum(confusion_matrix[1:,1:])/(np.sum(confusion_matrix[1:,1:]) + confusion_matrix[1,0] + confusion_matrix[2,0])
    Tn1 = confusion_matrix[0,0] + confusion_matrix[0,-1] + confusion_matrix[-1,0] + confusion_matrix[-1,-1]   
    specificity1 = Tn1/(Tn1 + confusion_matrix[0,1] + confusion_matrix[2,1]) 
    specificity2 = np.sum(confusion_matrix[:1,:1])/(np.sum(confusion_matrix[:1,:1]) + confusion_matrix[0,-1] + confusion_matrix[1,-1])

    specificity = [specificity0, specificity1, specificity2]

    assert specificity2 < 1 and specificity1 < 1 and specificity0 < 1 

    print("Classification report for classifier %s:\n%s\n"
        % (clf, metrics.classification_report(Y_test, predictions)))

    plt.figure()
    disp = metrics.plot_confusion_matrix(clf, X_test, Y_test)
    disp.figure_.suptitle("Confusion Matrix")
    print("Confusion matrix:\n%s" % disp.confusion_matrix)
    plt.savefig('../plots/'+dataset +'_mlp_confmatrix.png')
    
    return accuracy, sensitivity, specificity

def MultiLogisticRegression(data, targets, dataset, n_classes):
    print('APPLY LOGISTIC REGRESSION ...')
    X_train, Y_train = data[0], targets[0]
    X_val, Y_val = data[1], targets[1]
    X_test, Y_test = data[2], targets[2]
        
    clf = make_pipeline(StandardScaler(),LogisticRegression(random_state=0, penalty='l2', solver='liblinear'))

    clf.fit(X_train, Y_train)

    predictions = clf.predict(X_test)
    scores = clf.predict_proba(X_test)

    Y_test = list(Y_test) 

    assert len(predictions) == len(scores)
    assert len(scores) == len(Y_test)

    accuracy, class_report = multiclass_evaluate(clf, X_test, Y_test, np.array(predictions), np.array(scores), np.array(Y_test), n_classes, 'logisticregression'+dataset) 

    sensitivity = [class_report['0']['recall'], class_report['1']['recall'], class_report['2']['recall']]

    confusion_matrix = sklearn.metrics.confusion_matrix(y_true=Y_test, y_pred=predictions)

    print('Confusion Matrix', confusion_matrix)
    
    specificity0 = np.sum(confusion_matrix[1:,1:])/(np.sum(confusion_matrix[1:,1:]) + confusion_matrix[1,0] + confusion_matrix[2,0])
    Tn1 = confusion_matrix[0,0] + confusion_matrix[0,-1] + confusion_matrix[-1,0] + confusion_matrix[-1,-1]   
    specificity1 = Tn1/(Tn1 + confusion_matrix[0,1] + confusion_matrix[2,1]) 
    specificity2 = np.sum(confusion_matrix[:1,:1])/(np.sum(confusion_matrix[:1,:1]) + confusion_matrix[0,-1] + confusion_matrix[1,-1])

    specificity = [specificity0, specificity1, specificity2]

    assert specificity2 < 1 and specificity1 < 1 and specificity0 < 1 

    print("Classification report for classifier %s:\n%s\n"
    % (clf, metrics.classification_report(Y_test, predictions)))

    plt.figure()
    disp = metrics.plot_confusion_matrix(clf, X_test, Y_test)
    disp.figure_.suptitle("Confusion Matrix")
    print("Confusion matrix:\n%s" % disp.confusion_matrix)
    plt.savefig('../plots/'+dataset +'_logres_confmatrix.png')

    return accuracy, sensitivity, specificity


def GMM(data, targets, colors, dataset,target_names):
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

    ### GMM with Tied Covariance ### 

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

    accuracy, class_report = multiclass_evaluate(estimator, X_test, Y_test, np.array(predictions), np.array(scores), np.array(Y_test), n_classes, 'gmm'+dataset) 

    sensitivity = [class_report['0']['recall'], class_report['1']['recall'], class_report['2']['recall']]

    confusion_matrix = sklearn.metrics.confusion_matrix(y_true=Y_test, y_pred=predictions)

    print('Confusion Matrix', confusion_matrix)
    
    specificity0 = np.sum(confusion_matrix[1:,1:])/(np.sum(confusion_matrix[1:,1:]) + confusion_matrix[1,0] + confusion_matrix[2,0])
    Tn1 = confusion_matrix[0,0] + confusion_matrix[0,-1] + confusion_matrix[-1,0] + confusion_matrix[-1,-1]   
    specificity1 = Tn1/(Tn1 + confusion_matrix[0,1] + confusion_matrix[2,1]) 
    specificity2 = np.sum(confusion_matrix[:1,:1])/(np.sum(confusion_matrix[:1,:1]) + confusion_matrix[0,-1] + confusion_matrix[1,-1])

    specificity = [specificity0, specificity1, specificity2]

    assert specificity2 < 1 and specificity1 < 1 and specificity0 < 1 

    return accuracy, sensitivity, specificity


def apply_MultiML(data,targets, dataset, colors, target_names, n_classes): 

    scores = {'algorithm' : [], 'metric' : [], 'value' : [], 'dataset' : [], 'class': [] } 

    scores_ = {'sensitivity' : [], 'specificity' : [] , 'algorithm' : [], 'dataset' : [], 'class': []  }

    accuracy = {'algorithm' : [], 'value' : [], 'dataset': []}

    metric_names = ['sensitivity', 'specificity']
    
    ### KNN ### 
    # acc, sensitivity, specificity = MultiKNN(data, targets, dataset, n_classes)

    # metrics = [sensitivity, specificity]

    # for i, m in enumerate(metrics):
    #     for j in range(n_classes): 
    #         scores['algorithm'].append('KNN')
    #         scores['metric'].append(metric_names[i])
    #         scores['value'].append(metrics[i][j])
    #         scores['dataset'].append(dataset)
    #         scores['class'].append(target_names[j])
            
    # accuracy['algorithm'].append('KNN')
    # accuracy['value'].append(acc)
    # accuracy['dataset'].append(dataset)

    ## SVM ####
    acc, sensitivity, specificity = MultiSVM(data, targets, dataset, n_classes)

    metrics = [sensitivity, specificity]

    for i, m in enumerate(metrics):
        for j in range(n_classes): 
            scores['algorithm'].append('SVM')
            scores['metric'].append(metric_names[i])
            scores['value'].append(metrics[i][j])
            scores['dataset'].append(dataset)
            scores['class'].append(target_names[j])
            
    accuracy['algorithm'].append('SVM')
    accuracy['value'].append(acc)
    accuracy['dataset'].append(dataset)

    ### GMM ###
    # acc, sens, spec= GMM(data, targets, colors, dataset, target_names)

    # metrics = [sensitivity, specificity]

    # for i, m in enumerate(metrics):
    #     for j in range(n_classes): 
    #         scores['algorithm'].append('GMM')
    #         scores['metric'].append(metric_names[i])
    #         scores['value'].append(metrics[i][j])
    #         scores['dataset'].append(dataset)
    #         scores['class'].append(target_names[j])
            
    # accuracy['algorithm'].append('GMM')
    # accuracy['value'].append(acc)
    # accuracy['dataset'].append(dataset)

    ### Logistic Regression ###
    acc, sensitivity, specificity = MultiLogisticRegression(data, targets, dataset, n_classes)

    metrics = [sensitivity, specificity]

    for i, m in enumerate(metrics):
        for j in range(n_classes): 
            scores['algorithm'].append('LogRegress')
            scores['metric'].append(metric_names[i])
            scores['value'].append(metrics[i][j])
            scores['dataset'].append(dataset)
            scores['class'].append(target_names[j])
            
    accuracy['algorithm'].append('LogRegress')
    accuracy['value'].append(acc)
    accuracy['dataset'].append(dataset)

    ### Decision Tree ### 
    # acc, sensitivity, specificity = MultiDecision_Tree(data, targets, dataset, n_classes)

    # metrics = [sensitivity, specificity]

    # for i, m in enumerate(metrics):
    #     for j in range(n_classes): 
    #         scores['algorithm'].append('KNN')
    #         scores['metric'].append(metric_names[i])
    #         scores['value'].append(metrics[i][j])
    #         scores['dataset'].append(dataset)
    #         scores['class'].append(target_names[j])
            
    # accuracy['algorithm'].append('KNN')
    # accuracy['value'].append(acc)
    # accuracy['dataset'].append(dataset)

    ### MLP ### 
    acc, sensitivity, specificity = MultiMLP(data, targets, dataset, n_classes)

    metrics = [sensitivity, specificity]

    for i, m in enumerate(metrics):
        for j in range(n_classes): 
            scores['algorithm'].append('MLP')
            scores['metric'].append(metric_names[i])
            scores['value'].append(metrics[i][j])
            scores['dataset'].append(dataset)
            scores['class'].append(target_names[j])
            
    accuracy['algorithm'].append('MLP')
    accuracy['value'].append(acc)
    accuracy['dataset'].append(dataset)

    return accuracy, scores


def main(): 

    # datasets = [('../data/train_features_mlt.npy', '../data/train_targets_mlt.npy')]
    datasets = [] 
    # datasets.append(('../data/train_features_mlt_kde.npy', '../data/train_targets_mlt_kde.npy'))
    datasets.append(('../data/train_features_mlt_mse.npy', '../data/train_targets_mlt_mse.npy'))
    # datasets.append(('../data/train_features_mlt_kl.npy', '../data/train_targets_mlt_kl.npy'))
    #'multiclassbaseline' 
    # resultnames = [ 'multiclasslong_kde', 'multiclasslong_mse', 'multiclasslong_kl'] 
    # 'multiclasslong_kde',
    resultnames = ['multiclasslong_mse']


    for i, (dataset, resultname) in enumerate(zip(datasets,resultnames)): 

        feature, target = dataset
        print('Feature', feature)
        print('Target', target)
        X = np.load(feature, allow_pickle=True)
        Y = np.load(target, allow_pickle=True)
        print(X.shape, Y.shape)

        print('CN', Y.tolist().count(0))
        print('MCI', Y.tolist().count(1))
        print('Dementia', Y.tolist().count(2))

        # Sample MCI 
        X_MCI = X[Y==1]
        Y_MCI = Y[Y==1]
        X_Dementia = X[Y==2]
        Y_Dementia = Y[Y==2]
        X_CN = X[Y==0] 
        Y_CN = Y[Y==0]

        chosen_indices = []  
        while len(chosen_indices) < 350 :
            index = np.random.randint(low=0, high=X_MCI.shape[0]) 
            if index not in chosen_indices: 
                chosen_indices.append(index)
                

        X_MCI_sampled = X_MCI[chosen_indices]
        Y_MCI_sampled = Y_MCI[chosen_indices]
        
        
        chosen_indices = [] 
        while len(chosen_indices) < 350  :
            index = np.random.randint(low=0, high=X_CN.shape[0]) 
            if index not in chosen_indices: 
                chosen_indices.append(index)
        
        X_CN_sampled = X_CN[chosen_indices]
        Y_CN_sampled = Y_CN[chosen_indices]
        
        X = np.concatenate((X_MCI_sampled, X_Dementia, X_CN_sampled), axis=0)
        Y = np.concatenate((Y_MCI_sampled, Y_Dementia, Y_CN_sampled), axis=0)

        print('FINAL  DATA AFTER SAMPLING FOR BALANCED DATASETS')
        print(X.shape, Y.shape)
        print('CN', Y.tolist().count(0))
        print('MCI', Y.tolist().count(1))
        print('Dementia', Y.tolist().count(2))
 
        ### Stratify DATA ### 
        skf = StratifiedKFold(n_splits=20)
        # Only take the first fold.
        train_index, test_index = next(iter(skf.split(X,Y)))

        X_train = X[train_index]
        Y_train = Y[train_index]
        X_test = X[test_index]
        Y_test = Y[test_index]

        
        print('Train', X_train.shape, Y_train.shape)
        print('Test', X_test.shape, Y_test.shape)
        # count the classes in the train set 


        skf = StratifiedKFold(n_splits=2)
        val_index, _ = next(iter(skf.split(X_test,Y_test)))

        X_val = X_test[val_index]
        Y_val = Y_test[val_index]

        print('X_val', X_val.shape, Y_val.shape)

        colors = ['navy', 'turquoise' , 'darkorange']
        target_names = ['CN', 'MCI', 'Dementia']
        n_classes = len(np.unique(Y))

        X = [X_train, X_val, X_test]
        Y = [Y_train, Y_val, Y_test]

        accuracy, scores = apply_MultiML(data=X, targets=Y, dataset=resultname, colors=colors, target_names=target_names, n_classes=n_classes)


        print('Accuracy',accuracy)

        plt.figure()
        accuracy = pd.DataFrame(data=accuracy)
        sns.scatterplot(x='algorithm', y='value', data=accuracy)
        plt.xlabel('algorithms')
        plt.ylabel('accuracy')
        plt.xticks(rotation=45)
        plt.title('Overall Accuracy of ML Algorithms for Multiclass Classification')
        plt.savefig("../plots/all_algo_accuracy_"+ resultname+".png")

        
        scores = pd.DataFrame(data=scores)
        sns.relplot(x="algorithm", y="value", hue="metric", style="class", data=scores)
        plt.title('Comparison of ML Algorithms performance across datasets')
        plt.xlabel('algorithms')
        plt.ylabel('metric')
        plt.xticks(rotation=45)
        plt.show() 
        plt.savefig("../plots/sens_and_spec_per_class"+resultname + ".png")


if __name__ == "__main__":
    main()