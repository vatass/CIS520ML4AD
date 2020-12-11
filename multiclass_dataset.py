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

def check_longitudinal_dataset():

    with open('../longitudinal_dataset.pkl', 'rb') as f: 
        d = pickle.load(f)

    d = d['dataset']

    for i, sample in enumerate(d) : 

        print(i)
        feature, label = sample
        print(feature.shape, label)

def check_label(label):
    print(label)
    if label == 'CN->CN' : 
        label = 0 
    elif label == 'MCI->MCI' : 
        label = 1
    elif label == 'Dementia->Dementia': 
        label = 2 
    return label 
        
def define_dataset_multiclass_embeddings(): 


    with open('../longitudinal_dataset.pkl', 'rb') as f: 
        d = pickle.load(f)
    
    with open('../long_embeddings_kde.pickle', 'rb') as f: 
        kde = pickle.load(f)

    with open('../long_embeddings_triplet_KL.pickle', 'rb') as f: 
        kl = pickle.load(f)

    with open('../long_embeddings_triplet_mse_deep.pickle', 'rb') as f: 
        mse = pickle.load(f)

    d = d['dataset']
    kde = kde['dataset']
    kl = kl['dataset']
    mse = mse['dataset']

    new_kl_feature, new_kl_target = [], [] 
    new_kde_feature, new_kde_target = [], []  
    new_mse_feature, new_mse_target = [], []  
    true_labels, true_features = [],[]

    # CN = 0 
    # MCI = 1
    # DEM = 2 

    assert (len(d) == len(kde)) and (len(kl)==len(kde))

    # accept only MCI->MCI | Dementia->Dementia | CN->CN 
    accepted_labels = ['MCI->MCI', 'Dementia->Dementia', 'CN->CN']

    for i, sample in enumerate(kde) :
        feature, label = sample 
        
        if label in accepted_labels: 

            truefeature, truelabel = d[i]
            truelabel = check_label(truelabel)
            true_labels.append(truelabel)
            true_features.append(truefeature)

            # print(feature.shape)
            feature = feature.squeeze().cpu().numpy() 
            feature = np.expand_dims(feature, 0)
            # print(feature.shape)
            
            label = check_label(label)
            new_kde_feature.append(feature)
            new_kde_target.append(label)

            klfeature, kllabel = kl[i]
            # print('Initial KL feature', klfeature.shape)
            klfeature = klfeature.squeeze(0).squeeze(0) #.cpu().numpy() 
            # print('final kl feature', klfeature.shape)
            kllabel = check_label(kllabel)
            new_kl_feature.append(klfeature)
            new_kl_target.append(kllabel)

            msefeature, mselabel = mse[i]
            # print('Initial MSE feature', msefeature.shape)
            
            msefeature = msefeature.squeeze(0).squeeze(0)
            print('final mse', msefeature.shape)
            
            mselabel = check_label(mselabel)
            new_mse_feature.append(msefeature)
            new_mse_target.append(mselabel)

            print('MSE feature', msefeature.shape)
            print('KL feature', klfeature.shape)
            print('KDE feature', feature.shape)
          
            print(truelabel, mselabel, kllabel, label)
            assert mselabel == kllabel and kllabel == label and label==truelabel 
            

    print(len(new_kde_feature), len(new_mse_feature), len(new_kl_feature))
    assert len(new_kde_feature) == len(new_mse_feature) and len(new_kl_feature) == len(new_mse_feature)

    klfeatures = np.array(new_kl_feature)
    kltargets = np.array(new_kl_target)
    msefeatures = np.array(new_mse_feature)
    msetargets = np.array(new_mse_target)
    kdefeatures = np.array(new_kde_feature).squeeze(1)
    kdetargets = np.array(new_kde_target)
    print('KDE features',kdefeatures.shape)


    truelabels = np.array(true_labels)
    truefeatures = np.array(true_features)


    feature = '../data/train_features_mlt.npy'
    target = '../data/train_targets_mlt.npy'
    X = np.load(feature, allow_pickle=True)
    Y = np.load(target, allow_pickle=True)
    print(X.shape, Y.shape)

    print('CN', Y.tolist().count(0))
    print('MCI', Y.tolist().count(1))
    print('Dementia', Y.tolist().count(2))



    print('TRUE')
    print('CN', truelabels.tolist().count(0))
    print('MCI', truelabels.tolist().count(1))
    print('Dementia', truelabels.tolist().count(2))
    print('KL Targets')
    print('CN', kltargets.tolist().count(0))
    print('MCI', kltargets.tolist().count(1))
    print('Dementia', kltargets.tolist().count(2))
    print('KDE Targets')
    print('CN', kdetargets.tolist().count(0))
    print('MCI', kdetargets.tolist().count(1))
    print('Dementia', kdetargets.tolist().count(2))
    print('MSE Targets')
    print('CN', msetargets.tolist().count(0))
    print('MCI', msetargets.tolist().count(1))
    print('Dementia', msetargets.tolist().count(2))

    kdefeatures[0]

    ## Store Features and Target in .npy fils## 

    print('KDE features', kdefeatures.shape, len(kdetargets))
    np.save('../data/train_features_mlt_kde.npy', kdefeatures)
    np.save('../data/train_targets_mlt_kde.npy',kdetargets)
    
    print('KL features', klfeatures.shape, len(klfeatures))
    np.save('../data/train_features_mlt_kl.npy', klfeatures.squeeze(1))
    np.save('../data/train_targets_mlt_kl.npy',kltargets)
    
    print('MSE features', msefeatures.shape, len(msefeatures))
    np.save('../data/train_features_mlt_mse.npy', msefeatures.squeeze(1))
    np.save('../data/train_targets_mlt_mse.npy',msetargets)
    

def define_dataset_baseline_multiclass(): 
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
    
if __name__ == "__main__":

    
    # check_longitudinal_dataset()
    define_dataset_multiclass_embeddings() 
    # define_dataset_multiclass_svm() /

