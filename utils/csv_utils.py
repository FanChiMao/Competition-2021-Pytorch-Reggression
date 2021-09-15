import csv
import os
import pandas as pd
import statistics
from tqdm import tqdm


def write_csv(data=None, csv_path=None, save_name='result'):
    if not os.path.exists(csv_path):
        os.mkdir(csv_path)

    with open(csv_path + "/" + save_name + '.csv', 'w', newline='') as f:
        w = csv.writer(f)
        for i in range(len(data)):
            name = data[i][0]
            predict = data[i][1]
            w.writerow([name, predict])
    f.close()

def toIndependent_csv(csv_path=None, save_path=None, save_name='independent'):
    train = pd.read_csv(csv_path)   # 讀csv
    features = train.iloc[:, 1:-1].values   # 1~13特徵
    answers = train.iloc[:, 14].values  # 答案 (output)
    count = 1   # 輸出資料編號
    have_showed = []    # 儲存出現過的特徵(字串表示)
    have_showed_output = []
    final_indep_feat = []
    with open(save_path + "/" + save_name + '.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['No.', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'Output'])
        print('Get the independent data: ')
        for i, feat in enumerate(tqdm(features)):
            if str([*feat]) in have_showed:  # 這筆feature出現過
                list_number = have_showed.index(str([*feat]))
                have_showed_output[list_number].append(answers[i])  # 儲存重複特徵資料的不同輸出
            else:  # 這筆feature沒出現過
                have_showed.append(str([*feat]))  # 把沒出現過的特徵組儲存到出現過陣列
                final_indep_feat.append(feat)
                have_showed_output.append([answers[i]])  # 儲存輸出

        print('Write csv with calculated output:')
        for ii, out in enumerate(tqdm(have_showed_output)):

            # 輸出為平均值
            # final_output = statistics.mean(out)

            # 輸出為中位數
            # final_output = statistics.median(out)
            
            # 輸出為眾數
            final_output = statistics.mode(out)

            w.writerow([count, *final_indep_feat[ii], final_output])  # *是把array拆開成tuple
            count += 1
    f.close()


if __name__ == '__main__':
    wantTransCsvPath = '../csv_data/training/train_all.csv'
    saveCsvPath = '../csv_data/training/'
    save_name = 'independent_mean'
    toIndependent_csv(csv_path=wantTransCsvPath, save_path=saveCsvPath, save_name=save_name)
