import sys
sys.path.append('/envs/pytorch-HC-ZDgw_/lib/python3.8/site-packages/')

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import argparse



parser = argparse.ArgumentParser(description='Print Data')
parser.add_argument('-n', default=None, type=str)
args = parser.parse_args()

csv_path = '../../../projectA/train1/train1.csv'

data = pd.read_csv(csv_path)
result = data


for i, value in enumerate(data[args.n]):
    if 0 < i <= 3000:
        print(value)




