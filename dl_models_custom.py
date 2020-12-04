import numpy as np
import argparse
from argparse import ArgumentParser
import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import sys, os
from tqdm import tqdm
from dl_datasets import LongitudinalDiseaseClassification, LongitudinalDiseaseClassificationTestSet, TrajectoryDataset
from colors import bcolors 
import seaborn as sns
import matplotlib.pyplot as plt 

device="cpu"

class TemporalEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.progression_encoder = nn.Sequential(
            nn.LSTM(input_size=145, hidden_size=145, num_layers=1,bidirectional=False)
        )
        self.trajectory_encoder = nn.Sequential(input_size=145, hidden_size=145, num_layers=1, bidirectional=False)

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        progression_output, progression_hidden = self.progression_encoder(x)
        trajectory_output, trajectory_hidden = self.trajectory_encoder(trajectory_output)

        return progression_output, trajectory_output 

class ConvolutionalLongitudinalClassifier(nn.Module): 
    def __init__(self, classes):
        super().__init__()

        self.classes = classes 

        self.temporal_convolution = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(1,5)), 
        nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(1,5)), nn.BatchNorm2d(num_features=16), nn.ReLU() ,
        nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1,5)), nn.BatchNorm2d(num_features=8), nn.ReLU(),
        nn.Conv2d(in_channels=8, out_channels=1, kernel_size=(1,5)))

        self.linear = nn.Linear(in_features=129, out_features=classes)

    def forward(self, x): 

        if len(x.shape) == 3 and x.shape[0] == 1 :
            x = torch.unsqueeze(x,0)
        elif len(x.shape) == 3 and x.shape[0] !=1 and x.shape[1] == 1 :
            x = torch.reshape(x, (1, x.shape[1], x.shape[0], x.shape[2])) #n,c, h,w
        else :
            if x.shape[0] == 1 and x.shape[2] == 1 : 
                x = torch.reshape(x, (1,1,x.shape[1], x.shape[3]))


        temporal_maps = self.temporal_convolution(x)
        print('Temporal Feature Maps', temporal_maps.shape)
        vector = F.max_pool2d(temporal_maps, [temporal_maps.size(2), 1], padding=[0, 0])
        print('Feature Vector', vector.shape)
        embedding = self.linear(vector)
        print('Output', embedding.shape)

        return embedding.squeeze(0).squeeze(0)

class LongitudinalDiseaseClassifier(nn.Module):

    def __init__(self):
        super(LongitudinalDiseaseClassifier, self).__init__()
        self.temporal_model1 = nn.Sequential(
            nn.LSTM(input_size=145, hidden_size=150, num_layers=2,bidirectional=False)
        )

        self.classifier = nn.Sequential(nn.Linear(in_features=150, out_features=50), 
                                        nn.Linear(in_features=50, out_features=3)) 

    def forward(self, x):
        # LSTM input : (seq_len, batch, input_size)
        x = torch.reshape(x, (x.shape[1], x.shape[0], x.shape[2]))

        # print('In the classifier', x.shape, type(x))
         
        output, (hidden, context) = self.temporal_model1(x)
        # print('Output of LSTM', output.shape)

        y = self.classifier(output)

        # print('Hidden', hidden.shape)
        # print('Output', y.shape)

        out = y[-1,:,:]

        return  hidden, out


def train(batch_size, learning_rate, experiment_id,epochs): 

    ''' 
    hyperparams : dictionary with basic hyperparameters such as batch_size, learning_rate etc directly loaded from script's arguments
    '''

    val_every_n_epochs = 2

    report_and_plot_path = '/home/vtass/Desktop/experiments/' + str(experiment_id)  + '/'

    if not os.path.isdir(report_and_plot_path): 
        os.mkdir(path=report_and_plot_path)


    print(f"{bcolors.OKBLUE}===== DATASETS ====={bcolors.ENDC}")
    # data 
    dataset = LongitudinalDiseaseClassification(pckl_file='../longitudinal_dataset.pkl') 
    print('Train+Val Set', len(dataset))

    test_dataset = LongitudinalDiseaseClassificationTestSet(pckl_file='../longitudinal_dataset.pkl')
    print('Test Set', len(test_dataset))

    train, val = random_split(dataset, [850,25])
    print('Train', len(train))
    print('Val', len(val))
   
    train_loader = DataLoader(train, batch_size=1, num_workers=12)
    val_loader = DataLoader(val, batch_size=1, num_workers=12)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=12)


    # model 
    
    # model = LongitudinalDiseaseClassifier().float() 
    model = ConvolutionalLongitudinalClassifier(classes=3).float() 
    model = model.to(device)

    # loss 
    criterion = torch.nn.CrossEntropyLoss() 

    # optimizer 
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    print(f"{bcolors.OKBLUE}===== TRAIN BEGINS ====={bcolors.ENDC}")
    for i in range(epochs): 

        model.train() 
        train_loss = [] 
        for j, batch in tqdm(enumerate(train_loader)):

            feature,label = batch 

            feature = feature.float() 

            print('Input Shape', feature.shape)
            print('Label', label)

            # hidden, output = model(feature)
            out = model(feature)

            print('Output',out.shape)

            loss = criterion(out,label)

            train_loss.append(loss.item())
            loss.backward() 

            if j%batch_size != 0 and j!=0 : 
                optimizer.step()
                optimizer.zero_grad() 

        mean_train_loss = np.mean(train_loss)
        print(f"{bcolors.OKCYAN} Mean Train Loss {mean_train_loss}{bcolors.ENDC}")
        
        # plot train loss 
        sns.lineplot(x=range(len(train_loss)), y=train_loss)
        plt.savefig(report_and_plot_path + 'train_loss_epoch_' + str(i) + '.png')
                
        if i%val_every_n_epochs == 0 and i!=0 :
            model.eval() 

            val_loss = [] 
            for k, batch in enumerate(val_loader): 

                feature,label = batch 

                feature = feature.float() 

                with torch.no_grad(): 

                    # hidden,output = model(feature)
                    out = model(feature)

                    loss = criterion(out, label)

                    val_loss.append(loss.item())                    

            
            mean_val_loss = np.mean(val_loss)
            print(f"{bcolors.OKGREEN} Mean Val Loss {mean_val_loss}{bcolors.ENDC}")

            # plot val loss 
            sns.lineplot(x=range(len(val_loss)), y=val_loss)
            plt.savefig(report_and_plot_path + 'val_loss_epoch_' + str(i) + '.png')



if __name__ == "__main__" : 

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--experiment_id", type=str, default='longitud_convo_classification' )
    parser.add_argument("--epochs", type=int, default=5)

    args = parser.parse_args()

    train(**vars(args))







        


    



    





