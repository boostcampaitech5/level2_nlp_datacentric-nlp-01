import os
import argparse
import pandas as pd
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


def under_sampling(args):
    BASE_DIR = os.getcwd()
    DATA_DIR = os.path.join(BASE_DIR, '../data')
    df = pd.read_csv(os.path.join(DATA_DIR, args.file)) 
    count = Counter(df['target'])

    num = min(count.values())

    data_0 = df[df['target'] == 0][:num]
    data_1 = df[df['target'] == 1][:num]
    data_2 = df[df['target'] == 2][:num]
    data_3 = df[df['target'] == 3][:num]
    data_4 = df[df['target'] == 4][:num]
    data_5 = df[df['target'] == 5][:num]
    data_6 = df[df['target'] == 6][:num]

    data_list = [data_0, data_1, data_2, data_3, data_4, data_5, data_6]
    data_last = pd.concat(data_list)

    under_train = data_last.sort_values(by='ID', ascending=True)

    os.makedirs(DATA_DIR, exist_ok=True)
    under_train.to_csv(os.path.join(DATA_DIR, 'under_train.csv'), index=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, default="train.csv")
    args = parser.parse_args()
    
    under_train = under_sampling(args)

