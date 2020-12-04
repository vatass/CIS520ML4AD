import torch
import torch.nn as nn 
import torch.nn.functional as f
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import pickle 
import sys


def set_up_classification_dataset(dataset): 

    from sklearn import preprocessing

    l = dataset['dataset'] 
    data = [] 

    # CN : 0 
    # MCI : 1 
    # Dementia : 2 

    test_cn, test_mci, test_dem = [],[],[] 
    
    for k,(f, t) in enumerate(l): 

        # print(f.shape)
        ft = torch.from_numpy(f)
    
        # normalize 
        ft = preprocessing.scale(ft)

        if t == 'CN->CN': 
            
            target = [0,0,0]
            target[0] = 1
            
            # target = torch.from_numpy(np.array(target))
            # target = torch.from_numpy(np.array([0]))
            target = 0 
            if len(test_cn) < 10 : 
                test_cn.append((ft,target))
            else :
                data.append((ft,target))
        elif t == 'MCI->MCI': 
            
            target = [0,0,0]
            target[1] = 1
            # target = torch.from_numpy(np.array(target))
            target = 1
            # target = torch.from_numpy(np.array([1]))
            if len(test_mci) < 10 : 
                test_mci.append((ft,target))
            else: 
                data.append((ft,target))
            
        elif t == 'Dementia->Dementia': 
            target = [0,0,0]
            target[2] = 1
            # target = torch.from_numpy(np.array(target))

            # target = torch.from_numpy(np.array([2]))
            target=2 
            if len(test_dem) < 10 : 
                test_cn.append((ft,target))
            else : 
                data.append((ft,target))
 
    test_data = test_cn + test_mci + test_dem 

    


    return data, test_data

def create_triples_from_set(dataset):

    cn= []  
    mci = []
    dem = [] 
    data = [] 

    l = d['dataset']

    for k,(f, t) in enumerate(l): 
        print(f.shape)

        if t == 'CN->CN': 
            cn.append(f)
        elif t == 'MCI->MCI': 
            mci.append(f)
        elif t == 'Dementia->Dementia': 
            dem.append(f)

    print('CN->CN', len(cn))
    print('MCI->MCI', len(mci))
    print('Dementia->Dementia', len(dem))

    data_test = []

    # test_cn = cn[1:10]
    # test_mci = mci[1:10]
    # test_dem = dem[1:10]

    for i in range(100000): 

        index_cn = np.random.randint(low=0, high=len(cn))
        index_mci = np.random.randint(low=0, high=len(mci))
        index_dem = np.random.randint(low=0, high=len(dem))
        data.append((cn[index_cn], mci[index_mci,dem[index_dem]]))

    return data

class TrajectoryDataset(Dataset): 

    def __init__(self, pckl_file):
        """
        Args:
            csv_file (string): Path to the csv file with triplets
        """
        with open(pckl_file, 'rb') as f:
            d = pickle.load(f)

        self.triplets =  create_triples_from_set(dataset=d)

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):

        return self.triplets[idx] 

class LongitudinalDiseaseClassification(Dataset): 

    def __init__(self, pckl_file):
        """
        Args:
            pckl_file (string): Path to the csv file with triplets
        """
        with open(pckl_file, 'rb') as f:
            d = pickle.load(f)

        self.data,_ =  set_up_classification_dataset(dataset=d)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_, target = self.data[idx]
        return input_, target 

class LongitudinalDiseaseClassificationTestSet(Dataset): 

    def __init__(self, pckl_file):
        """
        Args:
            csv_file (string): Path to the csv file with triplets
        """
        with open(pckl_file, 'rb') as f:
            d = pickle.load(f)

        _, self.testdata =  set_up_classification_dataset(dataset=d)

    def __len__(self):
        return len(self.testdata)

    def __getitem__(self, idx):
        input_, target = self.testdata[idx]
        return input_, target 
