from argparse import ArgumentParser
import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl
import pickle 
import sys


def set_up_classification_dataset(dataset): 

    l = d['dataset'] 
    data = [] 

    # CN : 0 
    # MCI : 1 
    # Dementia : 2 

    for k,(f, t) in enumerate(l): 

        if t == 'CN->CN': 
            data.append(f,0)
        elif t == 'MCI->MCI': 
            data.append(f,1)
        elif t == 'Dementia->Dementia': 
            data.append(f,2)

    return data 

def create_triples_from_set(dataset):

    cn= []  
    mci = []
    dem = [] 
    data = [] 

    l = d['dataset']

    for k,(f, t) in enumerate(l): 

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
    with open(pickle_file, 'rb') as f:
        d = pickle.load(f)

    self.data, _ =  set_up_classification_dataset(dataset=d)

def __len__(self):
    return len(self.data)

def __getitem__(self, idx):
    return self.data[idx] 

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
    return self.testdata[idx] 

########################## MODELS ####################################### 

class KDELoss(nn.Module) : 

    pass 


class TemporalEncoder(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.progression_encoder = nn.Sequential(
            nn.LSTM(input_size=170, hidden_size=170, num_layers=1,bidirectional=False)
        )
        self.trajectory_encoder = nn.Sequential(input_size=170, hidden_size=170, num_layers=1, bidirectional=False)

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        progression_output, progression_hidden = self.progression_encoder(x)
        trajectory_output, trajectory_hidden = self.trajectory_encoder(trajectory_output)

        return progression_output, trajectory_output 

    def training_step(self, batch, batch_idx):
        
        # x : trajectory 1, y : trajectory : 2, z : no progression 

        x, y, z = batch() 
        
        progression_out_x, trajectory_out_x = self.TemporalEncoder(x) 
        progression_out_y, trajectory_out_y = self.TemporalEncoder(y)
        progression_out_z, trajectory_out_z = self.TemporalEncoder(z)

        progression_loss = -torch.nn.functional.mse_loss(progression_out_z, trajectory_out_x ) - torch.nn.functional.mse_loss(progression_out_z, trajectory_out_y)
        
        trajectory_loss = -torch.nn.functional.mse_loss(trajectory_out_y, trajectory_out_x )

        return progression_loss, trajectory_loss 

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

class LongitudinalDiseaseClassifier(pl.LightningModule):

    def __init__(self):
        super(LongitudinalDiseaseClassifier, self).__init__()
        self.encoder = nn.Sequential(
            nn.LSTM(input_size=170, hidden_size=170, num_layers=2,bidirectional=False)
        )
        self.classifier = nn.Sequential(self.Linear(in_features=170, out_features=3)) 

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        hidden, output = encoder(x)
        y = classifier(out)
        return  hidden, y

    def training_step(self, batch, batch_idx):
        
        # x : trajectory 1, y : trajectory : 2, z : no progression 

        data, target = batch() 
    
        hidden, out = self.LongitudinalDiseaseClassifier(data) 

        # out : seq_len, batch, num_directions * hidden_size  ->  s,1,170 

        # out : (N,C)
        # target : (N,1)
        loss = torch.nn.functional.cross_entropy(input=out, target=target)
        return loss 

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
 
def cli_main():
    pl.seed_everything(1234)

    # # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    dataset = LongitudinalDiseaseClassification(pckl_file='longitudinal_dataset.pkl') 
    print(type(dataset))
    print('Whole Dataset', len(dataset)) 
    sys.exit(0)
    test_dataset = LongitudinalDiseaseClassificationTestSet(pckl_file='longitudinal_dataset.pkl')

    train, val = random_split(dataset, [])

    train_loader = DataLoader(train, batch_size=1)
    val_loader = DataLoader(val, batch_size=1)
    test_loader = DataLoader(test_dataset, batch_size=1)

    # ------------
    # model
    # ------------
    model = LongitudinalDiseaseClassifier()

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, train_loader, val_loader)

    # ------------
    # testing
    # ------------
    result = trainer.test(test_dataloaders=test_loader)
    print(result)


if __name__ == '__main__':
    cli_main()

