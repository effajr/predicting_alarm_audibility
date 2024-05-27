"""
Creates a .csv file that splits the Development data in training and validation samples.
This code was used once to create a split that was kept the same during all the experiments.

"""

import pandas as pd
import random as rd
import numpy as np


def generate_validation_indices(conditions_csv_path, train_prop, output_csv_path):
    df = pd.read_csv(conditions_csv_path)
    indices = np.arange(len(df))

    train_len = np.ceil(train_prop * len(df)).astype(int)
    train_indices = np.array(rd.sample(indices.tolist(), train_len))
    valid_indices = np.setdiff1d(indices, train_indices)

    data_type = np.empty(len(df), dtype=np.dtype('<U6'))
    data_type[train_indices] = 'train'
    data_type[valid_indices] = 'valid'

    d = {'index': np.arange(len(df)), 'type': data_type}
    out = pd.DataFrame(data=d)
    out.to_csv(output_csv_path, index=False)

    return train_indices, valid_indices


if __name__ == '__main__':

    csv_path = '../data/annotations/dev/dev_conditions.csv'
    training_proportion = 0.8
    output_csv = '../data/annotations/dev/dev_train_valid_split.csv'
    # train_indices, valid_indices = generate_validation_indices(csv_path, training_proportion, output_csv)
