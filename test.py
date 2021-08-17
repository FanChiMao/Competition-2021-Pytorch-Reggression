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

model = MLP(n_inputs=13, hidden_layer1=128, hidden_layer2=128, hidden_layer3=64)


def test():
    torch.multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(description='Data Regression')

    parser.add_argument('--train_dir', default='./csv_data/training/train.csv', type=str)
    parser.add_argument('--test_dir', default='./csv_data/testing/test.csv', type=str)
    parser.add_argument('--result_dir', default='./csv_data/result/', type=str)
    parser.add_argument('--weights', default='./checkpoints/MLP/model/model_best.pth', type=str)
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
    score = 0
    max_single_score = 0
    print('===> Start testing~~')
    with torch.no_grad():
        for ii, data_test in enumerate(tqdm(test_loader, ncols=70, leave=False), 0):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            input_ = data_test[0].cuda()
            answer = data_test[1]
            file_names = data_test[2]

            predict = model(input_)
            predict = predict.cpu().numpy()

            for batch in range(len(predict)):
                results.append([file_names[batch].item(), predict[batch].item()])
                single_score = abs(predict[batch].item() - answer[batch].item())
                score += single_score

                if single_score > max_single_score:
                    max_single_score = single_score

    write_csv(data=results, csv_path=args.result_dir, save_name='results')

    x = score/len(test_dataset)
    y = max_single_score
    score_A = calculate_score_A(x)
    score_B = calculate_score_B(y)
    total_score = score_A + score_B
    print('===> Finish writing csv data!')

    print(f'''
    Result: 
    ----------------------------------
        Score A: {score_A}
        Score B: {score_B}
        Total:   {total_score}
    ''')

if __name__ == '__main__':
    test()
