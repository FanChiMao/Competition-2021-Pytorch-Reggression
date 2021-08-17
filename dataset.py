from torch.utils.data import Dataset
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

# dataset definition
class TrainDataset(Dataset):

    # load the dataset
    def __init__(self, train_path=None):
        # 讀csv檔併分割訓練資料欄位()與答案欄位()
        train_out = pd.read_csv(train_path)
        inputs = train_out.iloc[:, 1:14].values
        outputs = train_out.iloc[:, 14].values

        # feature scaling
        sc = StandardScaler()
        inputs_train = sc.fit_transform(inputs) # sc.fit_transform()
        outputs_train = outputs


        # 轉成張量(tensor)
        self.inputs_train = torch.tensor(inputs_train, dtype=torch.float32)
        self.outputs_train = torch.tensor(outputs_train, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.outputs_train)

    def __getitem__(self, idx):
        return self.inputs_train[idx], self.outputs_train[idx]


class ValDataset(Dataset):

    # load the dataset
    def __init__(self, train_path=None, val_path=None):
        # 讀csv檔併分割訓練資料欄位()與答案欄位()
        train_out = pd.read_csv(train_path)
        train_total = train_out.iloc[:, 1:14].values

        val_out = pd.read_csv(val_path)
        inputs = val_out.iloc[:, 1:14].values
        outputs = val_out.iloc[:, 14].values

        # feature scaling
        sc = StandardScaler()
        matrix = sc.fit_transform(train_total)
        inputs_train = sc.transform(inputs)
        outputs_train = outputs

        # 轉成張量(tensor)
        self.inputs_train = torch.tensor(inputs_train, dtype=torch.float32)
        self.outputs_train = torch.tensor(outputs_train, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.outputs_train)

    def __getitem__(self, idx):
        return self.inputs_train[idx], self.outputs_train[idx]
