import torch
import torch.nn.functional as F 
import torchvision.transforms as transforms
from torch import nn, optim


class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(102, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid()
            )

        def forward(self, x):
            return self.layers(x)
        



class MLP_dropout(nn.Module):
        def __init__(self, dropout_prob=0.5):
            super().__init__()
            self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(102, 128),
            nn.Dropout(dropout_prob),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.Dropout(dropout_prob),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.Dropout(dropout_prob),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid()
            )
            # self.initialize_weights()


        # def initialize_weights(self):
        #     for layer in self.layers:
        #         if isinstance(layer, nn.Linear):
        #             if layer == self.layers[-1]:
        #                 nn.init.xavier_uniform_(layer.weight)
        #             else:
        #                 nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')


        def forward(self, x):
            return self.layers(x)
        


class MLP_TanH(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(102, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Linear(64, 2),
            nn.Tanh()
            )

        def forward(self, x):
            return self.layers(x)
        

class MLP_TanH(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(102, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Linear(64, 2),
            nn.Tanh()
            )  
            self.initialize_weights()


        def initialize_weights(self):
            for layer in self.layers:
                if isinstance(layer, nn.Linear):
                    if layer == self.layers[-1]:
                        nn.init.xavier_uniform_(layer.weight)
                    else:
                        nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')

        def forward(self, x):
            return self.layers(x)
        