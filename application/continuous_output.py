"""
Computes the average

"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from predicting_alarm_audibility.scheme import *


def main(config):

    # Make DataLoaders
    model, dev_dl, eval_dl = make()

    # Select human responses
    pool_a = get_annotator_pool(dev_dl, eval_dl, data_type='eval', pool_id='pool_a')
    pool_a_indices = get_annotator_indices(eval_dl, pool_a, data_type='eval')
    # Get non-binarized APF values
    apf = np.mean(eval_dl.dataset.y[:, pool_a_indices[0]], axis=1)
    # Create a DataFrame with average human response for each listening condition
    human_outputs = pd.DataFrame({'alarm': eval_dl.dataset.alarm_ids,
                                  'background': eval_dl.dataset.background_id,
                                  'noise_level': eval_dl.dataset.noise_level,
                                  'snr': eval_dl.dataset.snr,
                                  'apf': apf})

    # Compute model outputs
    outputs_ = []
    directory = f"trained_models/{config['folder']}"
    for k in range(10):
        model_name = f'run_{k + 1}'
        model_ = load_weights(model, directory, model_name)
        _, output_ = inference(model_, eval_dl)
        outputs_.append(output_)
    avg_output = np.mean(np.array(outputs_), axis=0)
    # Create a DataFrame with average model output for each listening condition
    model_outputs = pd.DataFrame({'alarm': eval_dl.dataset.alarm_ids,
                                  'background': eval_dl.dataset.background_id,
                                  'noise_level': eval_dl.dataset.noise_level,
                                  'snr': eval_dl.dataset.snr,
                                  'avg_output': avg_output})

    # Display average outputs
    display_output(human_outputs, model_outputs)

    return human_outputs, model_outputs


def display_output(h_df, m_df):

    plt.figure('Continuous output')
    sns.lineplot(data=m_df, x='snr', y='avg_output', errorbar='se',
                 color='cornflowerblue', linewidth=1.2,
                 marker='s', markeredgecolor='cornflowerblue', markerfacecolor='None', markeredgewidth=1.2,
                 label='Model')
    sns.lineplot(data=h_df, x='snr', y='apf', errorbar='se',
                 color='cornflowerblue', linestyle='--', linewidth=1.2,
                 marker='o', markeredgecolor='cornflowerblue', markerfacecolor='None', markeredgewidth=1.2,
                 label='Human Baseline')
    plt.xlim([-25, 10])
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Continuous output / Average psychometric function')
    plt.show()
    plt.savefig('../figures/continuous_output.png', dpi=600)


def make():
    dev_ds_path = './features/dev/mel-spectrogram.h5'
    indices_csv_path = './data/annotations/dev/dev_train_valid_split.csv'
    dev_ds = Dataset(dev_ds_path, dev=True, split_indices_csv_path=indices_csv_path)
    dev_dl = DevDataLoader(dev_ds, 16)

    eval_ds_path = './features/eval/mel-spectrogram.h5'
    eval_ds = Dataset(eval_ds_path, dev=False, label_type='apf')
    eval_dl = EvalDataLoader(eval_ds, 16, training_mean=dev_dl.mean, training_std=dev_dl.std)

    model = CNN(in_channels=1)

    return model, dev_dl, eval_dl


if __name__ == "__main__":

    parameters = {'folder': 'paper'}
    h_out, m_out = main(parameters)
