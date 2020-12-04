import pickle
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import sklearn  
import sys 
from auxfunctions import select_baseline_data


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

    print('Unique Patients', len(list(df['participant_id'].unique())))

    print(df['Diagnosis'].unique())
    assert len(df['Diagnosis'].unique()) == 3 

    bdf = select_baseline_data(df=df)

    print(type(bdf), bdf.shape, len(bdf.index))

    ####### SELECT SOME SAMPLES FOR TEST OF THE WHOLE 2-LEVEL CLASSIFIER ########
    # drop_indices = [] 
    # for i in range(50): 
    # print(bdf.index)
    # sys.exit(0)
    # drop_indices = np.random.choice(bdf.index,50)
    # drop_indices.append(random_index)

    drop_indices = [0,    8,   22,   29,   33,   41,   53,   59,   64,   68,  9233, 9237, 9241, 9245, 9249, 9250, 9251, 9252, 9253, 9254]

    print('Before', bdf.shape)

    initial_bdf = bdf.copy() 

    for d in drop_indices: 

        if d in bdf.index: 
            print('to drop', d)
            bdf = bdf.drop(index=d)

    bdf_subset = bdf.copy() 
    print('After', bdf_subset.shape)
    
    for d in initial_bdf.index : 
        test_samples_df = initial_bdf.iloc[drop_indices]

    print('Selected test samples', test_samples_df.shape)  

    ##############################################################################


    Y1 = bdf['Diagnosis'].to_numpy() 
    X1 = bdf.filter(regex=("H_MUSE_Volume_*"))

    print('Features', X1.shape)
    print('Target', Y1.shape)
    print() 

    Y1[Y1=='Dementia' or Y1 =='MCI'] = 1 
    Y1[Y1=='CN'] = 0 
    # Y1[Y1=='MCI'] = 2 
    Y1=Y1.astype('int')

    ## Store Features and Target in .npy fils## 
    np.save('../data/first_level_features.npy', X1)
    np.save('../data/first_level_target.npy',Y1)

    Y2 = bdf['Diagnosis'].to_numpy() 
    X2 = bdf.filter(regex=("H_MUSE_Volume_*"))

    print('Features', X1.shape)
    print('Target', Y1.shape)
    print() 

    Y2[Y2=='Dementia' or Y2 =='MCI'] = 1 
 
    # Y1[Y1=='MCI'] = 2 
    Y2=Y2.astype('int')

    ## Store Features and Target in .npy fils## 
    np.save('../data/second_level_features.npy', X1)
    np.save('../data/second_level_target.npy',Y1)

def multiclass_SVM(): 


    # first develop an SVM for predicting Normal form Pathological 

    # Set Up Dataset for 1st SVM 





    # and then develop and SVM for predicting AD VS MCI 


    pass 


if __name__ == "__main__":


    define_dataset_multiclass_svm() 
    
    #define_dataset() 
    #multiclass_SVM() 
