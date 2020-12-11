import torch
import torch.nn as nn 
import torch.nn.functional as f
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import pickle 
import sys
import numpy as np


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
            # print('Dementia Patient Found !')
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

    l = dataset['dataset']

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


    # test_cn = cn[1:10]
    # test_mci = mci[1:10]
    # test_dem = dem[1:10]
    train_indices = [] 
    while len(train_indices) < 10000: 

        index_cn = np.random.randint(low=0, high=len(cn))
        index_mci = np.random.randint(low=0, high=len(mci))
        index_dem = np.random.randint(low=0, high=len(dem))
        tup = (index_cn, index_mci, index_dem)
        if tup not in train_indices: 
            train_indices.append(tup)
            data.append((cn[index_cn], mci[index_mci],dem[index_dem]))
    
    test_indices = [] 
    test_data = []
    while len(test_indices) < 1000: 
        index_cn = np.random.randint(low=0, high=len(cn))
        index_mci = np.random.randint(low=0, high=len(mci))
        index_dem = np.random.randint(low=0, high=len(dem))
        tup = (index_cn, index_mci, index_dem)
        if tup not in test_indices and tup not in train_indices:  
            test_indices.append(tup)
            test_data.append((cn[index_cn], mci[index_mci],dem[index_dem]))


    val_indices = np.random.choice(range(len(test_indices)), 100)
    val_data = [] 
    for v in val_indices:
        val_data.append(test_data[v])


    np.save('../data/train_triples', data)
    np.save('../data/val_triples', val_data)
    np.save('../data/test_triples', test_data)


    return data, test_data, val_data 

class TripletsDataset(Dataset): 

    def __init__(self, npyfile):
        """
        Args:
            npyfile (string): Path to the npy file with triplets
        """

        self.train_triplets = np.load(npyfile, allow_pickle=True)
        cn, mci, dem =  self.train_triplets[0]

        print('Train', len(self.train_triplets),type(self.train_triplets))
        print(cn.shape, mci.shape, dem.shape)


    def __len__(self):
        return len(self.train_triplets)

    def __getitem__(self, idx):
        cn, mci, dem = self.train_triplets[idx]
        return cn,mci,dem 

class TriplesTestDataset(Dataset): 
    def __init__(self, npyfile):
        """
        Args:
            npyfile (string): Path to the npy file with triplets
        """
        self.test_triplets = np.load(npyfile, allow_pickle=True)

    def __len__(self):
        return len(self.test_triplets)

    def __getitem__(self, idx):
        cn, mci, dem = self.test_triplets[idx]
        return cn,mci,dem 



class TriplesValDataset(Dataset): 
    def __init__(self, npyfile):
        """
        Args:
            npyfile (string): Path to the npy file with triplets
        """
        self.val_triplets = np.load(npyfile, allow_pickle=True)

    def __len__(self):
        return len(self.val_triplets)

    def __getitem__(self, idx):
        cn, mci, dem = self.val_triplets[idx]
        return cn,mci,dem 





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
        return torch.tensor(input_), target 

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



if __name__ == "__main__" : 

    with open('longitudinal_dataset.pkl', 'rb') as f:
        d = pickle.load(f)    

    _ =create_triples_from_set(d)



    

    