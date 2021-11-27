# Set Seedsimport sys
import sys
sys.path.append('/envs/pytorch-HC-ZDgw_/lib/python3.8/site-packages/')
import os
import torch

torch.backends.cudnn.benchmark = True

import utils
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse
from dataset import TrainDataset
import random
import numpy as np
from tqdm import tqdm
from model import MLP, MLP_3, MLP_drop
from warmup_scheduler import GradualWarmupScheduler
from sklearn.model_selection import train_test_split

random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

# Set Configuration
parser = argparse.ArgumentParser(description='Data Regression')

parser.add_argument('--n', default='MLP', type=str, help='network name')
parser.add_argument('--e', default=1000, type=int, help='training epochs')
parser.add_argument('--blr', default=0.001, type=float, help='bigin LR')
parser.add_argument('--flr', default=0.00001, type=float, help='finish LR')
parser.add_argument('--b', default=20000, type=int, help='batch size')
parser.add_argument('--vr', default=0.8, type=float, help='train_val_rate')
parser.add_argument('--pv', default=1, type=int, help='validation per epoch')
parser.add_argument('--train1', default='../../../projectA/train1/train1.csv',
                    type=str, help='training csv path')
parser.add_argument('--train2', default=None,
                            type=str, help='training2 csv path')
parser.add_argument('--rs', default=777, type=int, help='random state of train_val_split')
parser.add_argument('--save_dir', default='./checkpoints', type=str, help='model path')
parser.add_argument('--g', default=True, type=bool, help='GPU or not')

args = parser.parse_args()

# Model path direction
print('==> Build folder path')
start_epoch = 1
network_dir = os.path.join(args.save_dir, args.n)
utils.mkdir(network_dir)
save_dir = os.path.join(network_dir)
utils.mkdir(save_dir)
model_dir = os.path.join(network_dir, 'model')
utils.mkdir(model_dir)
train1 = args.train1
train2 = args.train2

# Model
print('==> Build MLP regressor')
# model = MLP(n_inputs=167, hidden_layer1=512, hidden_layer2=256, hidden_layer3=128)
# model = MLP_3(n_inputs=167, hidden_layer1=256, hidden_layer2=128)
model = MLP_drop(n_inputs=36, hidden_layer1=256, hidden_layer2=128, hidden_layer3=64)

# GPU device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if args.g:
    model.to(device=device) 

# Loss Function
criterion = nn.MSELoss()

# DataLoaders
print('==> Data preparation')
total_dataset = TrainDataset(train1, train2)
train_data, val_data = train_test_split(total_dataset, random_state=args.rs, 
                        train_size=args.vr)
train_loader = DataLoader(dataset=train_data, batch_size=args.b, shuffle=True, 
                        num_workers=0, drop_last=False, pin_memory=False)
val_loader = DataLoader(dataset=val_data, batch_size=len(val_data), shuffle=False,
                        num_workers=0, drop_last=False, pin_memory=False)

# Optimizer
print('==> Set LR strategy')
optimizer = torch.optim.Adam(model.parameters(), lr=args.blr, betas=(0.9, 0.999), eps=1e-8)

# Scheduler
warmup_epochs = 3
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.e - warmup_epochs, eta_min=float(args.flr))
scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
scheduler.step()

print(f'''==> Training details:
--------------------------------------------------------------------
    Network:            {args.n}
    Training data:      {len(train_data)}
    Validation data:    {len(val_data)}
    Start/End epochs:   {str(start_epoch) + '~' + str(args.e)}
    Batch sizes:        {args.b}
    Learning rate:      {args.blr}
    GPU:                {args.g}''')
print('--------------------------------------------------------------------')
print('==> Start Training :D')

best_val_loss = float('inf')
best_score = float('inf')
best_epoch_loss = 0
best_epoch_score = 0

for epoch in range(start_epoch, args.e + 1):
    epoch_loss = 0
    model.train()
    for i, data in enumerate(tqdm(train_loader, ncols=68, total=len(train_loader), leave=True), 0):
        for param in model.parameters():
            param.grad = None
        inputs = data[0].cuda()
        GT = data[1].cuda()

        out = model(inputs)

        predict_t = out[0]
        predict_x = out[1]
        predict_y = out[2]
        
        t_loss = criterion(predict_t, GT[0])
        x_loss = criterion(predict_x, GT[1])
        y_loss = criterion(predict_y, GT[2])

        loss = (t_loss + x_loss + y_loss)/3
        loss = torch.sqrt(loss)

        # optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        # validation (evaluation)
    if epoch % args.pv == 0:
        model.eval()
        score = 0
        for ii, data_val in enumerate(val_loader, 0):
            inputs = data_val[0].cuda()
            GT = data_val[1].cuda()

            with torch.no_grad():
                out = model(inputs)

            t = criterion(out[0], GT[0]) # angle
            x = criterion(out[1], GT[1]) # x
            y = criterion(out[2], GT[2])# y

            score += torch.sqrt((t+x+y)/3).item()

        if score < best_score:
            best_score = score
            best_epoch_score = epoch
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},os.path.join(model_dir, "best_score_model.pth"))

        print('  validation: RMSE Score {:.4f}\tBest_RMSE  {:.4f}  Epoch [{:4d}     ]'.format(score/len(val_loader), best_score/len(val_loader), best_epoch_score))
    print('  training:   RMSE Loss  {:.4f}\tLR     {:.6f}    Epoch [{:4d}/{}] '.format(epoch_loss/len(train_loader),scheduler.get_lr()[0], epoch, args.e))
    print("--------------------------------------------------------------------")
    torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, os.path.join(model_dir, "model_latest.pth"))
    scheduler.step()



print('==> Training Finished !!')

