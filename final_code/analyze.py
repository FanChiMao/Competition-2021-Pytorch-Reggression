import sys
sys.path.append('/envs/pytorch-HC-ZDgw_/lib/python3.8/site-packages/')
import pandas as pd

csv_path1 = '../../../projectA/train1/train1.csv'
csv_path2 = '../../../projectA/train2/train2.csv'
data = pd.read_csv(csv_path1)
data2 = pd.read_csv(csv_path2)
data = data.append(data2)

all_nan_col = data

l = all_nan_col.shape[0]
for col in all_nan_col.columns:
    print('Column name: ', col)
    if all_nan_col[col].isnull().sum() == l:
        print(' - have nan values: ', 'All data is NaN')
    elif all_nan_col[col].isnull().sum() == 0:
        print(' - have nan values:  ', all_nan_col[col].isnull().sum())
        print(' - different values: ', l - all_nan_col.duplicated(col).sum())
    else:
        print(' - different values: ', l - all_nan_col.duplicated(col).sum())
        print(' - different values: ', l - all_nan_col.duplicated(col).sum())
    print('--------------------------------------------')

print('Max t: ', data.loc[:, 'InspShift.t'].max())
print('Min t: ', data.loc[:, 'InspShift.t'].min())
print('Min x: ', data.loc[:, 'InspShift.x'].max())
print('Min x: ', data.loc[:, 'InspShift.x'].min())
print('Min y: ', data.loc[:, 'InspShift.y'].max())
print('Min y: ', data.loc[:, 'InspShift.y'].min())

