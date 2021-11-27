import sys
sys.path.append('/envs/pytorch-HC-ZDgw_/lib/python3.8/site-packages/')
import os
import argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import utils
from utils.csv_utils import write_csv
from dataset import TestDataset
from model import MLP_drop


print('==> Build model')
model = MLP_drop(n_inputs=36, hidden_layer1=256, hidden_layer2=128, hidden_layer3=64)

def demo():
    parser = argparse.ArgumentParser(description='Data Regression')
    parser.add_argument('--train_dir', default='../../../projectA/train1/train1.csv', type=str)
    parser.add_argument('--train_dir2', default='../../../projectA/train2/train2.csv', type=str)
    parser.add_argument('--test_dir', default='../../../projectA/test/test.csv', type=str)
    parser.add_argument('--result_dir', default='.', type=str)
    parser.add_argument('--save_name', default='110087_projectA_ans', type=str)
    parser.add_argument('--weights', default='./checkpoints/MLP_1/model/best_score_model.pth', type=str)
    parser.add_argument('--pr', default=True, type=bool, help='print result csv or not')

    args = parser.parse_args()
    
    print('==> Load weights to model')
    utils.load_checkpoint(model, args.weights)
    model.cuda()
    model.eval()

    train_dir = args.train_dir
    train_dir2 = args.train_dir2
    test_dir = args.test_dir
    
    print('==> Data preprocess')
    test_dataset = TestDataset(train_dir, train_dir2, test_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    results = []
    print('==> Start testing !!')

    with torch.no_grad():
        for ii, data_test in enumerate(tqdm(test_loader, ncols=68, leave=True), 0):
            input_ = data_test[0].cuda()
            SeqNo = data_test[1]

            predict = model(input_)
            predict = predict.cpu().numpy()

            for batch in range(len(predict)):
                results.append([SeqNo[batch].item(), predict[batch][0].item()/20, predict[batch][1].item(), predict[batch][2].item()])

    write_csv(data=results, csv_path=args.result_dir, save_name=args.save_name)
    print('==> Finish writing csv data!')
    print('----------------------------')

 

if __name__ == '__main__':
    demo()
