import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from numpy import pi, polymul
from scipy.signal import bilinear, lfilter


def read_audio(audio_path, target_fs=None):
    audio, fs = sf.read(audio_path)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs

    return fs, audio


def get_spl(x, fs, ponder='A'):

    def filter_a(x, fs):
        """
        Adapted from
        https://gist.githubusercontent.com/endolith/148112/raw/bff5182b7e65e9f3503bfde0f393b8851d587f39/A_weighting.py
        """
        f1 = 20.598997
        f2 = 107.65265
        f3 = 737.86223
        f4 = 12194.217
        a1000 = 1.9997

        nums = [(2 * pi * f4) ** 2 * (10 ** (a1000 / 20)), 0, 0, 0, 0]
        dens = polymul([1, 4 * pi * f4, (2 * pi * f4) ** 2],
                       [1, 4 * pi * f1, (2 * pi * f1) ** 2])
        dens = polymul(polymul(dens, [1, 2 * pi * f3]),
                       [1, 2 * pi * f2])
        b, a = bilinear(nums, dens, fs)

        y = lfilter(b, a, x)
        return y

    if ponder == 'lin':
        return 20 * np.log10(np.sqrt(np.mean(x**2)) / 2e-5)
    elif ponder == 'A':
        x_a = filter_a(x, fs)
        return 20 * np.log10(np.sqrt(np.mean(x_a**2)) / 2e-5)
    else:
        raise ValueError("Value for parameter 'ponder' must be 'lin' or 'A'.")


def get_gain(current_level, target_level):
    return np.sqrt(10 ** ((target_level-current_level)/10))


def make_audio_clip(background, alarm, fs, noise_level, snr, level_correction, alarm_onset, ponder='A'):

    background *= get_gain(get_spl(background, fs, ponder), noise_level)
    alarm *= get_gain(get_spl(alarm, fs, ponder), noise_level+snr-level_correction)

    padding_before = np.zeros(np.floor(alarm_onset * fs).astype(int) - 1)
    padding_after = np.zeros(background.shape[0] - alarm.shape[0] - padding_before.shape[0])
    padded_alarm = np.concatenate((padding_before, alarm, padding_after), axis=0)

    return background + padded_alarm


def get_standard_params(x):

    ndim_x = x.ndim

    if ndim_x == 2:
        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0)
    elif ndim_x == 3:
        mean = np.mean(x, axis=(0, 1))
        std = np.std(x, axis=(0, 1))
    elif ndim_x == 4:
        mean = np.mean(x, axis=(0, 2))
        std = np.std(x, axis=(0, 2))
    else:
        raise NotImplementedError('Input number of dimensions must be in (2, 3, 4)')

    return mean, std


def get_train_valid_indices(csv_path):

    df = pd.read_csv(csv_path)

    train_indices = df.loc[df['type'] == 'train'].index.values
    valid_indices = df.loc[df['type'] == 'valid'].index.values

    return train_indices, valid_indices


def get_mean_stderr(x):

    mean_x = np.mean(x)
    stderr_x = np.std(x) / np.sqrt(len(x))

    return mean_x, stderr_x
