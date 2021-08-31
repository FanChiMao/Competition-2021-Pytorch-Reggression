import torch.nn as nn

# model definition
class MLP(nn.Module):
    # define model elements
    def __init__(self, n_inputs, hidden_layer1, hidden_layer2, hidden_layer3):
        super(MLP, self).__init__()
        self.layer_1 = nn.Linear(n_inputs, hidden_layer1)
        self.act1 = nn.PReLU()
        self.layer_2 = nn.Linear(hidden_layer1, hidden_layer2)
        self.act2 = nn.PReLU()
        self.layer_3 = nn.Linear(hidden_layer2, hidden_layer3)
        self.act3 = nn.PReLU()
        self.layer_4 = nn.Linear(hidden_layer3, 1)
        self.act4 = nn.LeakyReLU(0.2)

    # forward propagate input
    def forward(self, x):
        x = self.act1(self.layer_1(x))
        x = self.act2(self.layer_2(x))
        x = self.act3(self.layer_3(x))
        x = self.act4(self.layer_4(x))
        return x



if __name__ == "__main__":
    m = MLP(13, 39, 39, 13)
    for idx, m in enumerate(m.modules()):
        print(idx, "-", m)