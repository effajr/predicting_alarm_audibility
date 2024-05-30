"""
Function that computes the human baseline performance.

"""
import pandas as pd
import numpy as np
from sklearn import metrics


def human_performance(csv_path, data_type, annotator_indices=None, strategy=None, metric=None, bin_threshold=0.5):

    # Read csv
    annotations = pd.read_csv(csv_path)

    # Restrict to selected annotators
    if annotator_indices is None:
        annotator_indices = np.arange(len(annotations.columns))
    annotations = annotations.iloc[:, annotator_indices]
    # Selected annotators
    annotators = annotations.columns

    # Compute metric
    performance = []
    for k in range(len(annotators)):
        annotator = annotators[k]
        responses_annotator = annotations.loc[:, annotator]
        responses_others = annotations.loc[:, [annot for annot in annotators if annot != annotator]]

        if data_type == 'dev' or strategy == 'mv':
            responses_others = (responses_others > bin_threshold).astype(int)
            responses_others = np.mean(responses_others, axis=1)
            if metric == 'roc_auc':
                performance.append(metrics.roc_auc_score((responses_others > bin_threshold).astype(int),
                                                         (responses_annotator > bin_threshold).astype(int)))
        elif data_type == 'eval' and strategy == 'apf':
            responses_others = np.mean(responses_others, axis=1)
            if metric == 'roc_auc':
                performance.append(metrics.roc_auc_score((responses_others > bin_threshold).astype(int),
                                                         responses_annotator))
        else:
            raise ValueError('Invalid input arguments!')

        if metric == 'roc_auc':
            pass
        elif metric in ['f1', 'precision', 'recall']:
            score = eval(f'metrics.{metric}_score')
            performance.append(score((responses_others > bin_threshold).astype(int),
                                     (responses_annotator > bin_threshold).astype(int)))
        else:
            raise ValueError("Input variable metric must be 'roc_auc', 'f1', 'precision' or 'recall'.")

    performance = np.array(performance)
    performance *= 100

    return performance
