import pickle
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import sklearn  
import sys 




def define_dataset():

    with open('../ADNI.pkl', 'rb') as f:
        d = pickle.load(f)

    df = d[d['Diagnosis'] =='MCI' or d['Diagnosis'] == 'Dementia' or d['Diagnosis']=='CN']

    print(df['Diagnosis'].unique())

    assert len(df['Diagnosis'].unique()) == 3 

    bdf = select_baseline_data(df=df)


    print('BASELINE MULTICALSS DATASET  : SET UP')
    print() 
    ### Convert Dataframe to Numpy Array 

    Y1 = bdf['Diagnosis'].to_numpy() 
    X1 = bdf.filter(regex=("H_MUSE_Volume_*"))

    print('Features', X1.shape)
    print('Target', Y1.shape)
    print() 

    Y1[Y1=='Dementia'] = 1 
    Y1[Y1=='CN'] = 0 
    Y1[Y1=='MCI'] = 2 
    Y1=Y1.astype('int')

    ## Store Features and Target in .npy fils## 
    np.save('../data/multiclass_features.npy', X1)
    np.save('../data/multiclass_target.npy',Y1)


    return X1, Y1 

    # l = d['dataset'] 
    # data = [] 

    # # CN : 0 
    # # MCI : 1 
    # # Dementia : 2 

    # data = {'diagnosis' : [], 'features' : []} 

    # cn, mci, dem = [], [], [] 

    # normal, disease = [], [] 

    # for k,(f, t) in enumerate(l): 
    #     print(f.shape)

    #     if t == 'CN->CN': 
    #         cn.append(f)
    #         data['diagnosis'].append('CN')
    #         data['features'].append(f)
    #         normal.append((f,0)) 

    #     elif t == 'MCI->MCI': 
    #         mci.append(f)
    #         data['diagnosis'].append('MCI')
    #         data['features'].append(f)
    #         disease.append

    #     elif t == 'Dementia->Dementia': 
    #         dem.append(f)
    #         data['diagnosis'].append('Dementia')
    #         data['features'].append(f)
    

    # print(len(cn), len(mci), len(dem))
    # # visualize the 3-classes 
    # plot_data = {'diagnosis' : [len(cn), len(mci), len(dem)], 'disease' : ['CN', 'MCI', 'Dementia'] }
    # plot_data = pd.DataFrame(data=plot_data, index=['CN', 'MCI', 'Dementia'] , columns=['diagnosis', 'disease'])

    # print(plot_data)
    # plot_data.plot.pie(y='diagnosis', figsize=(5, 5), colormap='summer')
    # plt.legend(loc='upper center')
    # plt.savefig("../plots/longitudinal_multiclasses.png")
    # plt.show() 

    # return cn,mci, dem


def multiclass_SVM(): 


    # first develop an SVM for predicting Normal form Pathological 

    # Set Up Dataset for 1st SVM 





    # and then develop and SVM for predicting AD VS MCI 


    pass 


if __name__ == "__main__":

    define_dataset() 
    multiclass_SVM() 
