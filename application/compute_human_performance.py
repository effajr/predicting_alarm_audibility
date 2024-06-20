# -*- coding: utf-8 -*-
"""
Computes the human baseline performance on the whole dataset as reported in the paper (see Tables 4 and 5).
The human baseline performance is computed on Development data (using MV labels) and on Evaluation data (using MV and
APF labels).

"""
import os
import sys
sys.path.append(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])
from scheme.human_baseline import human_performance
from scheme.utils import get_mean_stderr


if __name__ == "__main__":

    # Human baseline performance: Development data - Majority Voting
    auc_ = human_performance(csv_path='./data/annotations/dev/dev_labels.csv',
                             data_type='dev',
                             annotator_indices=None,
                             strategy='mv',
                             metric='roc_auc',
                             bin_threshold=0.5)
    f1_ = human_performance(csv_path='./data/annotations/dev/dev_labels.csv',
                            data_type='dev',
                            annotator_indices=None,
                            strategy='mv',
                            metric='f1',
                            bin_threshold=0.5)
    auc, f1 = get_mean_stderr(auc_), get_mean_stderr(f1_)
    print('Development data - Majority Voting')
    print(f"\t     AUROC: {auc[0]:.2f} +/- {auc[1]:.2f}, "
          f"    \t F1: {f1[0]:.2f} +/- {f1[1]:.2f}\n")

    # Human baseline performance: Evaluation data - Majority Voting
    auc_ = human_performance(csv_path='./data/annotations/eval/eval_labels_mv.csv',
                             data_type='eval',
                             annotator_indices=None,
                             strategy='mv',
                             metric='roc_auc',
                             bin_threshold=0.5)
    f1_ = human_performance(csv_path='./data/annotations/eval/eval_labels_mv.csv',
                            data_type='eval',
                            annotator_indices=None,
                            strategy='mv',
                            metric='f1',
                            bin_threshold=0.5)
    auc, f1 = get_mean_stderr(auc_), get_mean_stderr(f1_)
    print('Evaluation data - Majority Voting')
    print(f"\t     AUROC: {auc[0]:.2f} +/- {auc[1]:.2f}, "
          f"    \t F1: {f1[0]:.2f} +/- {f1[1]:.2f}\n")

    # Human baseline performance: Evaluation data - Average Psychometric Function
    auc_ = human_performance(csv_path='./data/annotations/eval/eval_labels_apf.csv',
                             data_type='eval',
                             annotator_indices=None,
                             strategy='apf',
                             metric='roc_auc',
                             bin_threshold=0.5)
    f1_ = human_performance(csv_path='./data/annotations/eval/eval_labels_apf.csv',
                            data_type='eval',
                            annotator_indices=None,
                            strategy='apf',
                            metric='f1',
                            bin_threshold=0.5)
    auc, f1 = get_mean_stderr(auc_), get_mean_stderr(f1_)
    print('Evaluation data - Average Psychometric Function')
    print(f"\t     AUROC: {auc[0]:.2f} +/- {auc[1]:.2f}, "
          f"    \t F1: {f1[0]:.2f} +/- {f1[1]:.2f}\n")
