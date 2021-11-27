import sys
sys.path.append('/envs/pytorch-HC-ZDgw_/lib/python3.8/site-packages/')
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, n_inputs, hidden_layer1, hidden_layer2, hidden_layer3):
        super(MLP, self).__init__()
        self.layer_1 = nn.Linear(n_inputs, hidden_layer1)
        self.act1 = nn.LeakyReLU(0.2)
        self.layer_2 = nn.Linear(hidden_layer1, hidden_layer2)
        self.act2 = nn.LeakyReLU(0.2)
        self.layer_3 = nn.Linear(hidden_layer2, hidden_layer3)
        self.act3 = nn.LeakyReLU(0.2)
        self.layer_4 = nn.Linear(hidden_layer3, 3)
        #self.act4 = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.act1(self.layer_1(x))
        x = self.act2(self.layer_2(x))
        x = self.act3(self.layer_3(x))
        #x = self.act4(self.layer_4(x))
        x = self.layer_4(x)
        return x

class MLP_drop(nn.Module):
    def __init__(self, n_inputs, hidden_layer1, hidden_layer2, hidden_layer3):
        super(MLP_drop, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(n_inputs, hidden_layer1),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_layer1, hidden_layer2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_layer2, hidden_layer3),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_layer3, 3),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.layer(x)
        return 10*x


class MLP_3(nn.Module):
    def __init__(self, n_inputs, hidden_layer1, hidden_layer2):
        super(MLP_3, self).__init__()
        self.layer_1 = nn.Linear(n_inputs, hidden_layer1)
        self.act1 = nn.LeakyReLU(0.2)
        self.layer_2 = nn.Linear(hidden_layer1, hidden_layer2)
        self.act2 = nn.LeakyReLU(0.2)
        self.layer_3 = nn.Linear(hidden_layer2, 3)

    def forward(self, x):
        x = self.act1(self.layer_1(x))
        x = self.act2(self.layer_2(x))
        x = self.layer_3(x)
        return x
            
