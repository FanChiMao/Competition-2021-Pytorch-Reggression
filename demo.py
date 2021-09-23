import os
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import utils
from utils.csv_utils import write_csv
from utils.score_utils import calculate_score_B, calculate_score_A
from dataset import TestDataset
from model import MLP

model = MLP(n_inputs=13, hidden_layer1=128, hidden_layer2=256, hidden_layer3=128)


def demo():
    torch.multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(description='Data Regression')

    parser.add_argument('--train_dir', default='./csv_data/training/independent_mean.csv', type=str)
    parser.add_argument('--test_dir', default='./csv_data/testing/2021test0831.csv', type=str)
    parser.add_argument('--result_dir', default='.', type=str)
    parser.add_argument('--save_name', default='result', type=str)
    parser.add_argument('--weights',
                        default='./pretrained.pth', type=str)
    parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    utils.load_checkpoint(model, args.weights)
    model.cuda()
    model.eval()

    train_dir = args.train_dir
    test_dir = args.test_dir

    test_dataset = TestDataset(train_dir, test_dir)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0)

    results = []
    print('===> Start testing~~')
    with torch.no_grad():
        for ii, data_test in enumerate(tqdm(test_loader, ncols=50, leave=True), 0):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            input_ = data_test[0].cuda()
            file_names = data_test[1]

            predict = model(input_)
            predict = predict.cpu().numpy()

            for batch in range(len(predict)):
                results.append([file_names[batch].item(), predict[batch].item()])


    write_csv(data=results, csv_path=args.result_dir, save_name=args.save_name)
    print('===> Finish writing csv data!')


if __name__ == '__main__':
    demo()
