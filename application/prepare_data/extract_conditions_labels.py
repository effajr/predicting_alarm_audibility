"""
Using the .csv file that contains all the annotator responses for each listening condition, this script separates the
unique listening conditions and individual annotator responses for each condition in separate .csv files.
This procedure is applied to both Development and Evaluation data.

"""

import pandas as pd
import numpy as np


def get_conditions_and_labels(filename, data_type):

    df = pd.read_csv(filename)
    df = df.sort_values(by=['annotator_id', 'noise_level', 'background_id', 'alarm_id', 'snr']).reset_index(drop=True)

    if data_type == 'eval':
        # restrict the table to unique listening conditions
        df_ = df[['alarm_id', 'background_id', 'noise_level', 'snr', 'level_correction_dB']].copy()
        conditions = df_.drop_duplicates(subset=['alarm_id', 'background_id', 'noise_level', 'snr',
                                                 'level_correction_dB']).copy()
        # add random alarm onsets for test data
        conditions['alarm_onset'] = 0.75 + (3.5 - 0.75) * np.random.random((len(conditions), 1))

    elif data_type == 'dev':
        df_ = df[['alarm_id', 'background_id', 'noise_level', 'snr', 'level_correction_dB', 'alarm_onset']].copy()
        conditions = df_.drop_duplicates(subset=['alarm_id', 'background_id', 'noise_level', 'snr',
                                                 'level_correction_dB', 'alarm_onset']).copy()
    else:
        raise ValueError("Variable 'data_type' must be 'dev' or 'eval'.")

    # get indices of the repetitions
    df_idx = (df_.groupby(df_.columns.tolist()).apply(lambda x: tuple(x.index)).reset_index(name='idx'))

    annotators = np.sort(df['annotator_id'].unique())

    if data_type == 'dev':
        labels = pd.DataFrame(columns=annotators)

        for index, row in conditions.iterrows():
            row = pd.DataFrame(row).T
            idx = np.array(row.merge(df_idx, how='inner',
                                     left_on=df_.columns.to_list(), right_on=df_.columns.to_list())['idx'][0])
            answers = df.loc[idx, ['annotator_id', 'clearly_audible']].sort_values(by='annotator_id')
            labels.loc[len(labels)] = answers['clearly_audible'].to_list()

        conditions = conditions.set_index(labels.index)
        return conditions, labels

    elif data_type == 'eval':
        labels_mv, labels_apf = pd.DataFrame(columns=annotators), pd.DataFrame(columns=annotators)

        for index, row in conditions.iterrows():
            row = pd.DataFrame(row).T
            idx = np.array(row.merge(df_idx, how='inner',
                                     left_on=df_.columns.to_list(), right_on=df_.columns.to_list())['idx'][0])
            answers = df.loc[idx, ['annotator_id', 'clearly_audible_mean', 'clearly_audible_pf']].\
                sort_values(by='annotator_id')
            labels_mv.loc[len(labels_mv)] = answers['clearly_audible_mean'].to_list()
            labels_apf.loc[len(labels_apf)] = answers['clearly_audible_pf'].to_list()

        conditions = conditions.set_index(labels_mv.index)
        return conditions, labels_mv, labels_apf


if __name__ == '__main__':

    dev_path = '../data/annotations/dev/annotation_compilation_dev.csv'
    dev_conditions, dev_labels = get_conditions_and_labels(dev_path, data_type='dev')
    dev_conditions.to_csv('../data/annotations/dev/dev_conditions.csv', index=False)
    dev_labels.to_csv('../data/annotations/dev/dev_labels.csv', index=False)

    eval_path = '../data/annotations/eval/annotation_compilation_eval.csv'
    eval_conditions, eval_labels_mv, eval_labels_apf = get_conditions_and_labels(eval_path, data_type='eval')
    eval_conditions.to_csv('../data/annotations/eval/eval_conditions.csv', index=False)
    eval_labels_mv.to_csv('../data/annotations/eval/eval_labels_mv.csv', index=False)
    eval_labels_apf.to_csv('../data/annotations/eval/eval_labels_apf.csv', index=False)
