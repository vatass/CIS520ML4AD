import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import sklearn  

def plot_multiple_classes(dataset): 

    labels = dataset['Diagnosis']
    sizes = df['values']
    colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral','cyan','lightpink']
    patches, texts = plt.pie(sizes, colors=colors, shadow=True, startangle=90, pctdistance=1.1, labeldistance=1.2)
    plt.legend(patches, labels, loc="best")
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def define_dataset(dataset):

    with open('../longitudinal_dataset.pkl', 'rb') as f:
    d = pickle.load(f)


    l = dataset['dataset'] 
    data = [] 

    # CN : 0 
    # MCI : 1 
    # Dementia : 2 

    data = {'diagnosis' : [], 'features' : []} 

    cn, mci, dementia = [], [], [] 

    for k,(f, t) in enumerate(l): 
        print(f.shape)

        if t == 'CN->CN': 
            cn.append(f)
            data['diagnosis'].append('CN')
        elif t == 'MCI->MCI': 
            mci.append(f)
            data['diagnosis'].append('MCI')
        elif t == 'Dementia->Dementia': 
            dem.append(f)
            data['diagnosis'].append('Dementia')
        
        data['features'].append('')

                






    pass 


def multiclass_SVM(): 
    pass 


if __name__ == "__main__":

    multiclass_SVM() 
