# -*- coding: utf-8 -*-
"""
Evaluates the trained models on Evaluation data using APF labels. As presented in the paper (see Table 6), ROC AUC and
F1-score are computed and reported with their mean values and standard error across 10 runs.
"""

import os
import sys
sys.path.append(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])
from scheme import *


def main():

    # Parameters
    nb_runs = 10
    batch_size = 16

    # Paths
    dev_ds_path = './features/dev/mel-spectrogram.h5'
    indices_csv_path = './data/annotations/dev/dev_train_valid_split.csv'
    eval_ds_path = './features/eval/mel-spectrogram.h5'

    # Make Datasets and Dataloaders
    dev_ds = Dataset(dev_ds_path, dev=True, split_indices_csv_path=indices_csv_path)
    dev_dl = DevDataLoader(dev_ds, batch_size)
    eval_ds = Dataset(eval_ds_path, dev=False, label_type='apf')
    eval_dl = EvalDataLoader(eval_ds, batch_size, training_mean=dev_dl.mean, training_std=dev_dl.std)
    # Make model
    model = CNN(in_channels=1)

    # Annotators: restrict to pool a
    pool_a = get_annotator_pool(dev_dl, eval_dl, data_type='eval', pool_id='pool_a')
    pool_a_indices = get_annotator_indices(eval_dl, pool_a, data_type='eval')

    # Compute AUC and F1 score
    auc_list, f1_list = [], []
    for run in range(nb_runs):
        run_name = f'run_{run + 1}'
        run_model = load_weights(model, directory='trained_models/paper', model_name=run_name)
        run_auc, run_f1 = evaluate(run_model, eval_dl, annotator_indices=pool_a_indices)
        auc_list.append(run_auc), f1_list.append(run_f1)

    auc_, f1_ = get_mean_stderr(auc_list), get_mean_stderr(f1_list)
    print(f'\tAUROC: {auc_[0]:.2f} ± {auc_[1]:.2f}\n\t   F1: {f1_[0]:.2f} ± {f1_[1]:.2f}')

    return auc_list, f1_list


if __name__ == "__main__":
    auc, f1 = main()
