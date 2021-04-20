import numpy as np
import torch
from torch import nn
import sys


class ValidationNet(nn.Module):

    def __init__(self, base_network: nn.Module, kdlayer: KernelDensityLayer):
        super().__init__()
        self.basenet = base_network
        self.kdlayer = kdlayer

    def set_kdpoints(self, new_points: torch.Tensor):
        self.kdlayer.set_points(new_points)

    def forward(self, x, return_latent=False, bandwidth=None):
        
        # print('Inside Validation Net')

        x = self.basenet(x)
        # print('Output', x.shape)
        # latent = x[-1,:,:] in LSTM 
        latent = x.squeeze(0)  # in Conv 
        if return_latent:
            
            return latent

        # latent = x[-1,:,:]

        logkde = self.kdlayer(latent, bandwidth)
        return -logkde

    def load_state_dict(self, state_dict, strict=True):
        self.set_kdpoints(state_dict["kdlayer.points"])
        super().load_state_dict(state_dict, strict=strict)




class LSTM_LongitudinalModeling(nn.Module): 
    def __init__(self,latent_dim: int = 32):
        super().__init__()

        self.latent_dim = latent_dim  

        self.temporal_model1 = nn.LSTM(input_size=145, hidden_size=150, num_layers=2)
                                                       
        self.temporal_model2 = nn.LSTM(input_size=150, hidden_size=self.latent_dim, num_layers=1)


    def forward(self, x): 

        # print('Input to Network', x.shape, type(x))
        out1, (hidden1,context1) = self.temporal_model1(x)
        # print('Intermediate input', out1.shape)
        out2, (hidden2, context2) = self.temporal_model2(out1)

        # print('Final Outpout 1', out2.shape)
        return out2 


class Convolutional_LongitudinalModeling(nn.Module): 
    def __init__(self,latent_dim: int = 64):
        super().__init__()

        # input shape 
        self.latent_dim = latent_dim  

        self.temporal_convolution = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(1,10)), 
        nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(1,10)), nn.BatchNorm2d(num_features=16), nn.ReLU() ,
        nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1,10)), nn.BatchNorm2d(num_features=32), nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,10)))

        self.linear = nn.Linear(in_features=109, out_features=latent_dim)

    def forward(self, x): 

        if len(x.shape) == 3 and x.shape[0] == 1 :
            # print('DO THE UNSQUEEZE')
            x = torch.unsqueeze(x,0)
        elif len(x.shape) == 3 and x.shape[0] !=1 and x.shape[1] == 1 :
            x = torch.reshape(x, (1, x.shape[1], x.shape[0], x.shape[2])) #n,c, h,w
        else :
            if x.shape[0] == 1 and x.shape[2] == 1 :
                # print('DO THE RESHAPE') 
                x = torch.reshape(x, (1,1,x.shape[1], x.shape[3]))

        # print('input', x.shape, type(x))


        # print('Corrected input shape', x.shape)

        temporal_maps = self.temporal_convolution(x)

        # print('Temporal Maps', temporal_maps.shape)
 

        vector = F.max_pool2d(temporal_maps, [temporal_maps.size(2), 1], padding=[0, 0])

        # print('Vector', vector.shape)
   
        embedding = self.linear(vector)

        # print('Embedding', embedding.shape)

        return embedding



class KernelDensityLayer(nn.Module):

    def __init__(self, bandwidth, points=None):
        super().__init__()
        self.bandwidth = bandwidth
        # self.points = points
        self.register_buffer("points", points)

    def set_points(self, new_points):
        self.register_buffer("points", new_points)

    def forward(self, x, bandwidth=None):
        # print('Inside KDE Layer')
        self.points = self.points.squeeze(1)
        x = x.squeeze(0)
        # print(self.points.shape, x.shape)
        if bandwidth is None:
            bandwidth = self.bandwidth

        logkde = torch_logkde(x, self.points, bandwidth)
        return logkde


def torch_logkde(query, points, bandwidth):

    #print('I am in the kde layer') 
    #print('query', query, len(query))
    #print('points', points, len(points), 'bandwidth', bandwidth) 
    #print(points.shape) 

    log_num_points = np.log(points.shape[0])  # log N 
    
    #print('log num points', log_num_points.shape)

    log_bandwidth = np.log(bandwidth)

    ndims = int(points.shape[1])

    #print('ndims', ndims) 

    sqrdists = torch.sum((query[:, None] - points[None, :])**2, dim=2)
    logkde = torch.logsumexp(-sqrdists / (2 * bandwidth**2), dim=1)
    logkde = logkde - log_num_points - ndims * (log_bandwidth + LOG_2PI * 0.5)
    
    #print('loglkde', logkde) 
    return logkde


def find_kdlayers(module):
    """Return all kd-layers used in a module."""
    kdlayers = filter(lambda x: isinstance(x, KernelDensityLayer),
                      module.modules())
    return list(kdlayers)

