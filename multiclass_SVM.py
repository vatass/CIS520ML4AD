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

    l = dataset['dataset'] 
    data = [] 

    # CN : 0 
    # MCI : 1 
    # Dementia : 2 

    test_cn, test_mci, test_dem = [],[],[] 
    




    pass 


def multiclass_SVM(): 
    pass 


if __name__ == "__main__":

    multiclass_SVM() 
