"""
Extracts the mel-spectrograms and creates the datasets in .h5 format for both Development and Evaluation data.

"""

import os
import numpy as np
import pandas as pd
import librosa
from scipy.signal import spectrogram
from predicting_alarm_audibility.scheme.utilities.utils import read_audio, make_audio_clip
import h5py


class MelSpectrogramExtractor:

    def __init__(self, fs, window, overlap, frequency_bins):
        self.window_size = window
        self.overlap = overlap
        self.ham_win = np.hamming(window)
        self.melW = librosa.filters.mel(sr=fs, n_fft=window, n_mels=frequency_bins, fmin=20., fmax=fs // 2).T

    def transform(self, audio):
        hamm_win = self.ham_win
        window_size = self.window_size
        overlap = self.overlap

        [_, _, x] = spectrogram(audio,
                                window=hamm_win,
                                nperseg=window_size,
                                noverlap=overlap,
                                detrend=False,
                                return_onesided=True,
                                mode='magnitude')

        x = x.T
        x = np.dot(x, self.melW)
        x = x.astype(np.float32)
        return x


def extract_features(conditions_csv_path, data_type, parameters,
                     dev_labels_csv_path=None, eval_labels_mv_csv_path=None, eval_labels_apf_csv_path=None):

    conditions = pd.read_csv(conditions_csv_path)

    fs = parameters['fs']
    freq_bins = parameters['freq_bins']
    window = parameters['window']
    overlap = parameters['overlap']
    seq_len = parameters['seq_len']
    extractor = MelSpectrogramExtractor(fs=fs, window=window, overlap=overlap, frequency_bins=freq_bins)

    hdf5_path = f"../features/{data_type}/{parameters['output_filename']}.h5"
    fd = os.path.dirname(hdf5_path)
    if not os.path.exists(fd):
        os.makedirs(fd)
    hf = h5py.File(hdf5_path, 'w')

    hf.create_dataset(name='feature',
                      shape=(0, seq_len, freq_bins),
                      maxshape=(None, seq_len, freq_bins),
                      dtype=np.float32)

    identifiers = []
    id_alarms = []
    id_backgrounds = []
    noise_levels = []
    snrs = []
    level_corrections = []
    alarm_onsets = []

    for idx in conditions.index:
        identifiers.append('id' + str(idx))
        id_alarms.append(conditions['alarm_id'][idx])
        id_backgrounds.append(conditions['background_id'][idx])
        noise_levels.append(conditions['noise_level'][idx])
        snrs.append(conditions['snr'][idx])
        level_corrections.append(conditions['level_correction_dB'])
        alarm_onsets.append(conditions['alarm_onset'][idx])

        audio_dir = f'../data/audio/{data_type}/'
        alarm_name, background_name = conditions['alarm_id'][idx]+'.wav', conditions['background_id'][idx]+'.wav'
        alarm_onset, noise_level = conditions['alarm_onset'][idx], conditions['noise_level'][idx]
        snr, level_correction = conditions['snr'][idx], conditions['level_correction_dB'][idx]
        _, alarm = read_audio(os.path.join(audio_dir, 'alarms/') + alarm_name, target_fs=fs)
        _, background = read_audio(os.path.join(audio_dir, 'backgrounds/') + background_name, target_fs=fs)
        clip = make_audio_clip(background, alarm, fs, noise_level, snr, level_correction, alarm_onset)
        feature = extractor.transform(clip)
        hf['feature'].resize((idx + 1, seq_len, freq_bins))
        hf['feature'][idx] = feature

        print(f'\r   Features extracted for: {idx + 1}/{conditions.shape[0]} conditions', end='')

    if data_type == 'dev':
        clearly_audible = pd.read_csv(dev_labels_csv_path)
        id_annotators = clearly_audible.columns.to_numpy().astype(int)
        labels = clearly_audible.loc[:].to_numpy()
        hf.create_dataset(name='label', data=labels, dtype=int)
    elif data_type == 'eval':
        clearly_audible_mv = pd.read_csv(eval_labels_mv_csv_path)
        clearly_audible_apf = pd.read_csv(eval_labels_apf_csv_path)
        id_annotators = clearly_audible_mv.columns.to_numpy().astype(int)
        labels_mv = clearly_audible_mv.loc[:].to_numpy()
        labels_apf = clearly_audible_apf.loc[:].to_numpy()
        hf.create_dataset(name='label_mv', data=labels_mv, dtype=np.float32)
        hf.create_dataset(name='label_apf', data=labels_apf, dtype=np.float32)
    else:
        raise ValueError("Variable 'data_type' must be 'dev' or 'eval'.")

    hf.create_dataset(name='identifier', data=[s.encode() for s in identifiers], dtype='S20')
    hf.create_dataset(name='alarm_id', data=[s.encode() for s in id_alarms], dtype='S20')
    hf.create_dataset(name='background_id', data=[s.encode() for s in id_backgrounds], dtype='S20')
    hf.create_dataset(name='noise_level', data=noise_levels, dtype=np.float32)
    hf.create_dataset(name='snr', data=snrs, dtype=np.float32)
    hf.create_dataset(name='level_correction', data=level_corrections, dtype=np.float32)
    hf.create_dataset(name='alarm_onset', data=alarm_onsets, dtype=np.float32)
    hf.create_dataset(name='annotators', data=id_annotators, dtype=int)
    hf.close()


if __name__ == '__main__':

    config = {'fs': 44100,
              'window': 1024,
              'overlap': 512,
              'freq_bins': 64,
              'seq_len': 472,
              'output_filename': 'mel-spectrogram'}

    conditions_path = '../data/annotations/dev/dev_conditions.csv'
    labels_path = '../data/annotations/dev/dev_labels.csv'
    extract_features(conditions_path, data_type='dev', parameters=config, dev_labels_csv_path=labels_path)

    conditions_path = '../data/annotations/eval/eval_conditions.csv'
    labels_mv_path = '../data/annotations/eval/eval_labels_mv.csv'
    labels_apf_path = '../data/annotations/eval/eval_labels_apf.csv'
    extract_features(conditions_path, data_type='eval', parameters=config,
                     eval_labels_mv_csv_path=labels_mv_path, eval_labels_apf_csv_path=labels_apf_path)
