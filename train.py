import yaml
import os
import torch

from utils import network_parameters

torch.backends.cudnn.benchmark = True

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import dataset
import random
from dataset import TrainDataset, ValDataset
import time
import numpy as np
import utils
from model import MLP
from tqdm import tqdm
from tensorboardX import SummaryWriter

## Set Seeds
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

## Load yaml configuration file
with open('training.yaml', 'r') as config:
    opt = yaml.safe_load(config)
Train = opt['TRAINING']

## Model and log path direction
print('==> Build folder path')
start_epoch = 1
network_dir = os.path.join(Train['SAVE_DIR'], Train['Network'])
utils.mkdir(network_dir)
save_dir = os.path.join(network_dir)
utils.mkdir(save_dir)
model_dir = os.path.join(network_dir, 'models')
utils.mkdir(model_dir)
log_dir = os.path.join(network_dir, 'log')
utils.mkdir(log_dir)
train_dir = Train['TRAIN_DIR']
val_dir = Train['VAL_DIR']

writer = SummaryWriter(log_dir=log_dir, filename_suffix=f'_log')

# GPU device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model
print('==> Build model')
model = MLP(n_inputs=13, hidden_layer1=128, hidden_layer2=128, hidden_layer3=64)
if Train['GPU']:
    model.to(device=device)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=Train['LR'])

# Loss function
criterion = nn.L1Loss()

# DataLoaders
print('==> Data preparation')
train_dataset = TrainDataset(train_dir)
train_loader = DataLoader(dataset=train_dataset, batch_size=Train['BATCH'],
                          shuffle=True, num_workers=0, drop_last=False, pin_memory=False)
val_dataset = ValDataset(train_dir, val_dir)
val_loader = DataLoader(dataset=val_dataset, batch_size=Train['BATCH'], shuffle=False, num_workers=0,
                        drop_last=False, pin_memory=False)

print(f'''==> Training details:
------------------------------------------------------------------------------
    Network:            {Train['Network']}
    Training data:      {len(train_dataset)}
    Validation data:    {len(val_dataset)}
    Start/End epochs:   {str(start_epoch) + '~' + str(Train['EPOCH'] + 1)}
    Batch sizes:        {Train['BATCH']}
    Learning rate:      {Train['LR']}
    GPU:                {Train['GPU']}''')
print('------------------------------------------------------------------------------')

# train
best_val_loss = 10000
best_epoch = 0
for epoch in range(start_epoch, Train['EPOCH'] + 1):
    epoch_loss = 0
    model.train()
    for i, data in enumerate(tqdm(train_loader, ncols=50, total=len(train_loader), leave=False), 0):

        for param in model.parameters():
            param.grad = None

        inputs = data[0].cuda()
        GT = data[1].cuda()
        out = model(inputs)
        loss = criterion(out, GT)

        # optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(" Epoch [{:3d}/{}]: \tLoss: {:.4f}\t".format(epoch, Train['EPOCH'] + 1, epoch_loss / len(train_loader)))
    print("------------------------------------------------------------------")
    torch.save({'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
                }, os.path.join(model_dir, "model_latest.pth"))

    # validation (evaluation)
    if epoch % Train['VAL_AFTER_EVERY'] == 0:
        model.eval()
        epoch_val_loss = 0
        for ii, data_val in enumerate(val_loader, 0):
            inputs = data_val[0].cuda()
            GT = data_val[1].cuda()
            with torch.no_grad():
                out = model(inputs)
            val_loss = criterion(out, GT)
            epoch_val_loss += val_loss.item()

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_epoch = epoch
            torch.save({'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, os.path.join(model_dir, "model_best.pth"))

        print("[epoch %d Loss: %.4f --- best_epoch %d Best_loss %.4f]" % (
            epoch, epoch_val_loss / len(val_loader), best_epoch, best_val_loss / len(val_loader)))

        torch.save({'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, os.path.join(model_dir, f"model_epoch_{epoch}.pth"))
        writer.add_scalar('val/loss', epoch_val_loss / len(val_loader), epoch)
    writer.add_scalar('train/loss', epoch_loss / len(train_loader), epoch)
writer.close()
