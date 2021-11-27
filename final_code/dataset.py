import sys
sys.path.append('/envs/pytorch-HC-ZDgw_/lib/python3.8/site-packages/')
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from encode.target_encode import TargetEncoder


def OHEncode(df, colNames):
    for col in colNames:
        if( df[col].dtype == np.dtype('object')):
            dummies = pd.get_dummies(df[col],prefix=col)
            df = pd.concat([dummies, df],axis=1)
             # drop the encoded column
            df.drop([col],axis = 1 , inplace=True)
    return df

def FQEncode(df, colNames):
    for col in colNames:
        count = df[col].value_counts()
        for i in range(len(count.index)):
            df = df.replace(count.index[i], count.values[i])
    return df

def LBEncode(df, colNames):
    for col in colNames:
        df[col] = df[col].replace('nan', 0)
        df = df.replace('LeftRtHead2', 1)
        df = df.replace('LeftRtHead2_2', 2)
        df = df.replace('LeftRtHead4', 3)
    return df

def TGEncode(df, colNames, test_df=None, ans=None):
    if ans is not None:
        mean_target = ans.mean(axis=1)
    else:
        mean_target = df.iloc[:, -3:].mean(axis=1)
    encoder = TargetEncoder(cols=colNames)
    df = encoder.fit_transform(df, mean_target)
    if test_df is not None:
        return df, encoder.transform(test_df)
    return df

certain_col = ['BondingTime', 'BondingDate', 'BondingDateTime']
single_val_col = ['M_Adj_p0.x', 'M_Adj_p0.y', 'M_Adj_p1.x', 'M_Adj_p1.y']
fusion_col = ['GlassDieOffsetX', 'GlassDieOffsetY', 'InspTemperature', 'InspSamplingLearnAdjY', 'RoomTemperature']
class TrainDataset(Dataset):
    def __init__(self, train_1=None, train_2=None):
        print('------------------------------------------------------------------')        
        train =pd.read_csv(train_1)
        print('  Train_1 data columns:                                      ', train.shape[1])
        
        if train_2 is not None:
            train2 = pd.read_csv(train_2)
            print('  Train_2 data columns:                                      ', train2.shape[1])
            train = train.append(train2)
            # train.reset_index(inplace=True)
            print('  Concat data numbers:                                   ', train.shape[1])

        # Remove whole NaN
        rm_train = train
        rm_train = rm_train.dropna(how='all', axis=1)
        print('  Remove the columns which all of values are NaN:             ', rm_train.shape[1])
 
        rm_train = rm_train.drop(columns=certain_col)
        print('  Remove the columns which we consider useless for training:  ', rm_train.shape[1])

        rm_train = rm_train.drop(columns=single_val_col)
        print('  Remove the columns which has single values, zero:           ', rm_train.shape[1])
    
        # Condition remove columns
        l = rm_train.shape[0]
        for col in rm_train.columns:
            if rm_train[col].isnull().sum() > 0.5 * l:
                rm_train = rm_train.drop(columns=col)

        print('  Remove the columns which has lots of NaN values (> 0.5):    ', rm_train.shape[1])
        
        # One hot-encode
        #rm_OHE = oneHotEncode(rm_train, ['WaferID', 'HeadName', 'Recipe', 'GlassID'])
        #print('  OneHotEncode columns:', rm_OHE.shape[1])
            
        # Encode category----------------------------------------------------------
        # Label Encode
        rm_train = LBEncode(rm_train, ['HeadName'])

        # Frequency Encode
        # rm_train = FQEncode(rm_train, ['GlassID', 'Recipe', 'WaferID']) # 'WaferID'

        # Target Encode
        rm_train = TGEncode(rm_train, ['Recipe', 'GlassID']) #,'WaferID']) # , 'WaferID' have value: 11410
        print('  Encode the category values to numerical format:             ', rm_train.shape[1])
        
        # Fuse features
        rm_train = rm_train.drop(columns=fusion_col)
        rm_train['GlassDieOffsetT'] = rm_train['GlassDieOffsetT'].replace(0, 1).fillna(0)
        rm_train['InspHumidity'] = rm_train['InspHumidity'].replace(3, 1)
        rm_train['InspSamplingLearnAdjX'] = rm_train['InspSamplingLearnAdjX'].replace(-0.0012, -0.5).fillna(-1)
        rm_train['RoomHumidity'] = rm_train['RoomHumidity'].fillna(-1)
        print('  Supervisely simplify the relative columns:                  ', rm_train.shape[1])

        rm_train = rm_train.fillna(rm_train.mean())
        print('  Fill NaN values to average:                                 ', rm_train.shape[1])

        rm_train['InspShift.t'] = rm_train['InspShift.t']*20
        print('                                                                  ') 
        print('  Sequence numbers:                                            ', 1)
        inputs = rm_train.iloc[:, 1:-3].values
        print('  Input features:                                             ', rm_train.iloc[:, 1:-3].shape[1])
        outputs = rm_train.iloc[:, -3:].values
        print('  Output answers:                                              ', rm_train.iloc[:, -3:].shape[1])
        print('------------------------------------------------------------------')
        
        sc = StandardScaler()
        inputs_train = sc.fit_transform(inputs)
        outputs_train = outputs

        self.inputs_train = torch.tensor(inputs_train, dtype=torch.float32)
        self.outputs_train = torch.tensor(outputs_train, dtype=torch.float32)

    def __len__(self):
        return len(self.outputs_train)
    def __getitem__(self, idx):
        return self.inputs_train[idx], self.outputs_train[idx]

class TestDataset(Dataset):
    def __init__(self, train_1=None, train_2=None, test_path=None):
        train = pd.read_csv(train_1)
        test = pd.read_csv(test_path)
        test['InspShift.t'] = 0
        test['InspShift.x'] = 0
        test['InspShift.y'] = 0
        file_names = test.iloc[:, 0].values

        if train_2 is not None:
            train2 = pd.read_csv(train_2)
            train = train.append(train2)

        train_ans = train.iloc[:, -3:]
        rm_train = train.iloc[:, 1:-3]
        rm_test = test

        train_num = rm_train.shape[0]
        cat_data = pd.concat([rm_train, rm_test], axis=0)
        dropna_data = cat_data.dropna(axis=1, how='all')
        rm_train = dropna_data.iloc[0:train_num, :]
        rm_test = dropna_data.iloc[train_num:, :]

        rm_train = rm_train.drop(columns=certain_col)
        rm_test = rm_test.drop(columns=certain_col)
        rm_train = rm_train.drop(columns=single_val_col)
        rm_test = rm_test.drop(columns=single_val_col)
        
        l = dropna_data.shape[0]
        #rm_train = rm_train.drop(columns='WafeID')
        for col in dropna_data.columns:
            
            if dropna_data[col].isnull().sum() > 0.5 * l:
                rm_train = rm_train.drop(columns=col)
        rm_test = rm_test.drop(columns=['MeanAdj.t','MeanAdj.x','MeanAdj.y', 'PickTopRecDieAdj_DieOnPreciser.t', 'PickTopRecDieAdj_DieOnPreciser.x',
            'PickTopRecDieAdj_DieOnPreciser.y','PickTopRecSearchAdj_DieOnPreciser.t', 'PickTopRecSearchAdj_DieOnPreciser.x', 'PickTopRecSearchAdj_DieOnPreciser.y',
            'SearchResultAtAppZAdj0.x', 'SearchResultAtAppZAdj0.y','SearchResultAtAppZAdj1.x', 'SearchResultAtAppZAdj1.y','SearchResultAtAppZAdj2.x', 'SearchResultAtAppZAdj2.y','SearchResultAtAppZAdj3.x','SearchResultAtAppZAdj3.y','SearchResultAtAppZAdj4.x','SearchResultAtAppZAdj4.y','SearchResultAtAppZAdj5.x','SearchResultAtAppZAdj5.y','SearchResultAtAppZAdj6.x','SearchResultAtAppZAdj6.y','TuneData.x','TuneData.y', 'WaferCol', 'WaferRow','InspLearnAdjX','InspLearnAdjY',
            'Sliding0.x','Sliding0.y', 'Sliding1.x','Sliding1.y','Sliding2.x','Sliding2.y','Sliding3.x','Sliding3.y','Sliding4.x','Sliding4.y','Sliding5.x','Sliding5.y',
            'Sliding6.x','Sliding6.y',
            'BonderPlaceTime', 'BondingCurrentPressure','FastModeTuneDataTheta','WaferID'])
        
        rm_test = rm_test.drop(columns=col)

        rm_train, rm_test = TGEncode(rm_train, ['Recipe', 'GlassID'], rm_test, train_ans)

        rm_train = LBEncode(rm_train, ['HeadName'])
        rm_test  = LBEncode(rm_test,  ['HeadName'])
        
        rm_train = rm_train.drop(columns=fusion_col)
        rm_test = rm_test.drop(columns=fusion_col)
        rm_train['GlassDieOffsetT'] = rm_train['GlassDieOffsetT'].replace(0, 1).fillna(0)
        rm_test['GlassDieOffsetT'] = rm_test['GlassDieOffsetT'].replace(0, 1).fillna(0)
        rm_train['InspHumidity'] = rm_train['InspHumidity'].replace(3, 1)
        rm_test['InspHumidity'] = rm_test['InspHumidity'].replace(3, 1)
        rm_train['InspSamplingLearnAdjX'] = rm_train['InspSamplingLearnAdjX'].replace(-0.0012, -0.5).fillna(-1)
        rm_test['InspSamplingLearnAdjX'] = rm_test['InspSamplingLearnAdjX'].replace(-0.0012, -0.5).fillna(-1)
        rm_train['RoomHumidity'] = rm_train['RoomHumidity'].fillna(-1)
        rm_test['RoomHumidity'] = rm_test['RoomHumidity'].fillna(-1)
        
        rm_train = rm_train.drop(columns='WaferID')
        rm_test = rm_test.drop(columns='SeqNo')
        rm_train = rm_train.fillna(rm_train.mean())
        rm_test = rm_test.fillna(rm_train.mean())
        
        inputs = rm_test.values

        sc = StandardScaler()
        matrix = sc.fit_transform(rm_train)
        inputs_test = sc.transform(inputs)
        names = file_names

        self.inputs_test = torch.tensor(inputs_test, dtype=torch.float32)
        self.file_name = torch.tensor(names, dtype=torch.int32).view(-1, 1)

    def __len__(self):
        return len(self.inputs_test)

    def __getitem__(self, idx):
        return self.inputs_test[idx], self.file_name[idx]


