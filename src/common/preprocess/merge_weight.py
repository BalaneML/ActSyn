'''
個人単位で重みづける
'''

import pandas as pd
import numpy as np


DATASET_PATH = "../../data/processed/dataset_merged.csv"
WEIGHT_PATH  = "../../data/raw/NHTS2022/perv2pub.csv"
OUT_PATH     = "../../data/processed/weighted_dataset.csv"


def merge_weight(dataset_path: str, weights_path: str):
    '''重み列追加処理'''
    dataset = pd.read_csv(dataset_path)
    weights = pd.read_csv(
        weights_path, usecols=['HOUSEID', 'PERSONID', 'WTPERFIN']
    )  # TODO: 平日weightや土日weightも検討 (WTPERFIN5D, WTPERFIN2D)
    print(f'パーソン数: {len(dataset)}')
    print(f'weight数: {len(weights)}')

    merged = pd.merge(dataset, weights, on=['HOUSEID', 'PERSONID'], how='inner')

    base_cols = list(dataset.columns)
    new_order = base_cols[:2] + ['WTPERFIN'] + base_cols[2:]
    weighted_dataset = merged[new_order]
    print(f'マージ後の行数: {len(weighted_dataset)}')
    return weighted_dataset


def main():
    weighted_dataset = merge_weight(DATASET_PATH, WEIGHT_PATH)
    weighted_dataset.to_csv(OUT_PATH, index=False)
    print(weighted_dataset)
    print(weighted_dataset.shape)
    print('Dataset is saved to ' + OUT_PATH)
    

if __name__ == "__main__":
    main()
