import pickle
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import sklearn  
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

import sklearn.metrics as metrics
import sys 
from auxfunctions import select_baseline_data, evaluate
import sklearn.svm as svm 
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline

def define_dataset_multiclass_svm(): 
    '''
    Prepare the dataset accordingly for the first-level binary classification  
    Annotate the CN as healthy and the Dementia/MCI as pathological 
    CN = 0 
    Dementia/MCI = 1 
    '''

    with open('../ADNI.pkl', 'rb') as f: 
        d = pickle.load(f)

    print('Unique Patients', len(list(d['participant_id'].unique())))

    df = d[(d['Diagnosis'] == 'MCI') | (d['Diagnosis'] == 'Dementia') | (d['Diagnosis'] == 'CN')]

    # df = d[(d['Diagnosis'] == 'Dementia') | (d['Diagnosis'] == 'CN')]
    
    print('Unique Patients', len(list(df['participant_id'].unique())))

    print(df['Diagnosis'].unique())
    assert len(df['Diagnosis'].unique()) == 3

    bdf = select_baseline_data(df=df)

    print(type(bdf), bdf.shape, len(bdf.index))

    ####### SELECT SOME SAMPLES FOR TEST OF THE WHOLE 2-LEVEL CLASSIFIER ########
    # select the first 20 test samples 
    testsamples=100

    drop_indices = list(bdf.index[0:testsamples ])
    print(drop_indices)

    print('Before', bdf.shape)
    print('Indexes in bf', len(list(bdf.index)))

    initial_bdf = bdf.copy() 
    print('Shape of initial_bdf', initial_bdf.shape)
    print('Indexes in initial bf', len(list(bdf.index)))

    test_samples = [] 

    for d in drop_indices: 

        if d in bdf.index: 
            # print('to drop', d)
            bdf = bdf.drop(index=d)

    print('After bdf', bdf.shape, len(list(bdf.index)))

    bdf_subset = bdf.copy() 

    print('bdf_subset', bdf_subset.shape, len(list(bdf_subset.index)))

    #### SAVE THE BDF SUBSET ####
    Y1 = bdf_subset['Diagnosis'].to_numpy() 
    X1 = bdf_subset.filter(regex=("H_MUSE_Volume_*")).to_numpy()

    assert X1.shape[0] == Y1.shape[0]
  
    ## Drop Half of the diagnosis that are Dementia and Half of the diagnosis that are MCI 
    ## So as to make that train dataset for the 1-level classifier balanced

    print('Features', X1.shape)
    print('Target', Y1.shape)
    print() 

    Y1[Y1=='Dementia'] = 2
    Y1[Y1=='MCI'] = 1
    Y1[Y1=='CN'] = 0 
    Y1=Y1.astype('int')

    ## Store Features and Target in .npy fils## 
    np.save('../data/train_features_mlt.npy', X1)
    np.save('../data/train_targets_mlt.npy',Y1)
    ######


    for k in range(len(drop_indices)): 
        tmp = initial_bdf.iloc[k]
        print(type(tmp))
        test_samples.append(tmp)

    print('Selected test samples', len(test_samples))  
    
    test_dataframe = pd.concat(test_samples, axis=1).T
    
    print('Test Dataframe', test_dataframe.shape)
    # print(test_dataframe.head(10))

    Ytest = test_dataframe['Diagnosis'].to_numpy() 
    Xtest = test_dataframe.filter(regex=("H_MUSE_Volume_*")).to_numpy() 

    print(Ytest.shape) 

    Ytest[Ytest=='Dementia'] = 2
    Ytest[Ytest=='CN'] = 0 
    Ytest[Ytest=='MCI'] = 1
    Ytest=Ytest.astype('int')

    np.save('../data/test_features.npy', Xtest)
    np.save('../data/test_target.npy',Ytest)
    '''
    ##############################################################################

    print('TIME TO CURATE THE SETS', bdf_subset.shape)

    Y1 = bdf_subset['Diagnosis'].to_numpy() 
    X1 = bdf_subset.filter(regex=("H_MUSE_Volume_*")).to_numpy()

    assert X1.shape[0] == Y1.shape[0]
  
    ## Drop Half of the diagnosis that are Dementia and Half of the diagnosis that are MCI 
    ## So as to make that train dataset for the 1-level classifier balanced

    print('Features', X1.shape)
    print('Target', Y1.shape)
    print() 

    Y1[Y1=='Dementia'] = 1
    Y1[Y1=='MCI'] = 1 
    Y1[Y1=='CN'] = 0 
    # Y1[Y1=='MCI'] = 2 
    Y1=Y1.astype('int')

    ## Store Features and Target in .npy fils## 
    np.save('../data/first_level_features.npy', X1)
    np.save('../data/first_level_target.npy',Y1)

    #### LEVEL 2 DATASET 
    #remove all the CN from the bdf_subset 

    bdf_subset = bdf_subset[bdf_subset['Diagnosis']!='CN']

    print(bdf_subset['Diagnosis'].unique())

    Y2 = bdf_subset['Diagnosis'].to_numpy() 
    X2 = bdf_subset.filter(regex=("H_MUSE_Volume_*")).to_numpy()

    print('Features', X2.shape)
    print('Target', Y2.shape)
    print() 

    assert X2.shape[0] == Y2.shape[0]

    Y2[Y2=='MCI'] = 0 
    Y2[Y2=='Dementia'] = 1 
    Y2=Y2.astype('int')

    ## Store Features and Target in .npy fils## 
    np.save('../data/second_level_features.npy', X2)
    np.save('../data/second_level_target.npy',Y2)
    '''

def multiclass_SVM(): 

    #######  First develop an SVM for predicting Normal form Pathological  ########
    # Set Up Dataset for 1st SVM 

    traindata = np.load('../data/first_level_features.npy')
    traintargets = np.load('../data/first_level_target.npy')

    assert traindata.shape[0] == traintargets.shape[0]

    Xtrain1, Xval1, Ytrain1, Yval1 = train_test_split(traindata, traintargets, test_size=0.05, random_state=42) 

    ### Apply SVM 
    print('APPLY SVM ...')
    print('With Data of Shape', Xtrain1.shape)

    ### Grid Search to some parameters ####
    params_grid = [{'kernel': ['rbf'], 'gamma': [1e-3],
                     'C': [1]},
                    {'kernel': ['linear'], 'C': [10]}]

    # 'C' :  [1, 10, 100, 1000]
    # 'gamma': [1e-3, 1e-4]

    # svm_model = GridSearchCV(SVC(), params_grid, cv=2)
    final_model1 = svm.SVC()

    final_model1.fit(Xtrain1, Ytrain1)

    # View the accuracy score
    # print('Best score for training data:', svm_model.best_score_,"\n") 

    # # View the best parameters for the model found using grid search
    # print('Best C:',svm_model.best_estimator_.C,"\n") 
    # print('Best Kernel:',svm_model.best_estimator_.kernel,"\n")
    # print('Best Gamma:',svm_model.best_estimator_.gamma,"\n")

    # final_model1 = svm_model.best_estimator_

    ### test the model on the 1st task only 

    ### From the general test data set up the test data 

    Xtest1 = np.load('../data/test_features.npy', allow_pickle=True)
    Ytest1 = np.load('../data/test_target.npy', allow_pickle=True)

    assert Xtest1.shape[0] == Ytest1.shape[0]

    # CN = 0 
    # MCI = 1 
    # DEM = 2 
    # NORMAL = 0 (CN)  PATHOLOGICAL = 1 (MCI,DEM)

    Ytestlist = [] 
    # print('Before', Ytest1)
    for i in range(Ytest1.shape[0]):
        if Ytest1[i] == 2 :  
            Ytestlist.append(1)
        else: 
            Ytestlist.append(Ytest1[i])
        
    # print('After', Ytestlist)

    Ytestarr = np.expand_dims(np.array(Ytestlist),1) 

    print(Xtest1.shape, Ytestarr.shape, type(Xtest1), type(Ytestarr))


    predictions = final_model1.predict(Xtest1)
    scores = final_model1.decision_function(Xtest1)

    acc, prec, rec, sens, spec = evaluate(classifier=final_model1,
                                          X_test=Xtest1,
                                          Y_test=Ytestarr,
                                          prediction=np.array(predictions), 
                                          score=np.array(scores),
                                          groundtruth=Ytestarr,
                                          algo= 'svm') 
    print('Test Accuracy, Precision, Recall', acc, prec, rec)
    print()

    ######## and then develop and SVM for predicting AD VS MCI  #########
    # Set Up Dataset for 2nd SVM 

    traindata = np.load('../data/second_level_features.npy')
    traintargets = np.load('../data/second_level_target.npy')

    Xtrain2,  Xval2, Ytrain2,  Yval2 = train_test_split(traindata, traintargets) 

    params_grid = [{'kernel': ['rbf'], 'gamma': [1e-3],
                     'C': [1]},
                    {'kernel': ['linear'], 'C': [10]}]

    # svm_model = GridSearchCV(SVC(), params_grid, cv=2)
    final_model2 = svm.SVC()

    # final_model2 = SVC() 
    final_model2.fit(Xtrain2, Ytrain2)

    # View the accuracy score
    # print('Best score for training data:', svm_model.best_score_,"\n") 

    # # View the best parameters for the model found using grid search
    # print('Best C:',svm_model.best_estimator_.C,"\n") 
    # print('Best Kernel:',svm_model.best_estimator_.kernel,"\n")
    # print('Best Gamma:',svm_model.best_estimator_.gamma,"\n")

    # final_model2 = svm_model.best_estimator_

    ###### test the model on the 2nd task task only 
    print('Test on 2nd Task')

    Xtest2 = np.load('../data/test_features.npy', allow_pickle=True)
    Ytest2 = np.load('../data/test_target.npy', allow_pickle=True)

    print(Xtest2.shape, Ytest2.shape)

    # CN = 0 
    # MCI = 1 
    # DEM = 2 
    # NORMAL = 0 (CN)  PATHOLOGICAL = 1 (MCI,DEM)

    # Keep only the 1, 2 and then convert the MCI -) 0 and Dementia to 1 

    Ytestlist, index_to_pop = [], [] 
    print('Before', Ytest2)
    for i in range(Ytest2.shape[0]):
        if Ytest2[i] == 0 : 
            index_to_pop.append(i)
        
        elif Ytest2[i] == 1 :  
            Ytestlist.append(0)

        elif Ytest2[i] == 2: 
            Ytestlist.append(1)

    print('Indexes to remove', len(index_to_pop))
    Xtestlist = [] 

    for j in range(Xtest2.shape[0]): 

        if j in index_to_pop: 
            pass
        else : 
            print(Xtest2[j,:].shape)
            Xtestlist.append(np.expand_dims(Xtest2[j,:], 0))


    Xtest_filt = np.concatenate(Xtestlist, axis=0)

    print('After', Ytestlist)
    print(Xtest_filt.shape)

    Ytestarr = np.expand_dims(np.array(Ytestlist), 1) 

    assert Ytestarr.shape[0] == Xtest_filt.shape[0]

    predictions = final_model2.predict(Xtest_filt)
    scores = final_model2.decision_function(Xtest_filt)

    acc, prec, rec, sens, spec = evaluate(classifier=final_model2,
                                         X_test=Xtest_filt,
                                         Y_test=Ytestarr,
                                         prediction=np.array(predictions),
                                         score=np.array(scores),
                                         groundtruth=Ytestarr,
                                         algo='svm') 
    print('Test Accuracy, Precision, Recall', acc, prec, rec)
    print()


    ###### TEST THE FINAL 2-LEVEL SVM SYSTEM ####### 

    Xtest2 = np.load('../data/test_features.npy', allow_pickle=True)
    Ytest2 = np.load('../data/test_target.npy', allow_pickle=True)

    final_predictions = [] 

    for sample, gt in zip(Xtest2, Ytest2) : 

        # use the 1st level SVM 
        
        print('Level 1')
        print('sample', sample.shape)
        sample = np.expand_dims(sample,0)
        prediction = final_model1.predict(sample)
        print('Groundtruth',gt )
        print('Prediction', prediction[0])
        score = final_model1.decision_function(sample)
        print('Relative Distance from the Decision Boundary', score)
    
        # scores is the relative distance from the boundaries 

        if prediction[0] == 0 : 
            print('Patient Normal')
            final_pred = 0 
        elif prediction[0] == 1 : 
            print('Patient Pathological')

            predictions = final_model2.predict(sample)
            scores = final_model2.decision_function(sample)

            if prediction[0] == 0 : 
                print('Patient with MCI')
                final_pred = 1 
            elif prediction[0] == 1 : 
                print('Patient with Dementia')
                final_pred = 2 

        final_predictions.append(final_pred)


    report = classification_report(y_true=Ytest2, y_pred=final_predictions)
    print(report)

def sklearn_SVM_multiclass_classif(): 
    pass
    print('Load Train')
    trainf = np.load('../data/train_features_mlt.npy')
    traint = np.load('../data/train_targets_mlt.npy')
    print('Load Test')
    X_test = np.load('../data/test_features.npy', allow_pickle=True)
    Y_test = np.load('../data/test_target.npy', allow_pickle=True)

    X_train, X_val, Y_train, Y_val = train_test_split(trainf, traint, test_size=0.01, random_state=42) 

    classifier = make_pipeline(StandardScaler(), SVC(gamma='scale'))

    classifier.fit(X_train, Y_train)

    predicted = classifier.predict(X_test)

    print("Classification report for classifier %s:\n%s\n"
        % (classifier, metrics.classification_report(Y_test, predicted)))

    disp = metrics.plot_confusion_matrix(classifier, X_test, Y_test)
    disp.figure_.suptitle("Confusion Matrix")
    print("Confusion matrix:\n%s" % disp.confusion_matrix)

    plt.show()


if __name__ == "__main__":


    define_dataset_multiclass_svm() 
    sklearn_SVM_multiclass_classif()
    # multiclass_SVM() 
