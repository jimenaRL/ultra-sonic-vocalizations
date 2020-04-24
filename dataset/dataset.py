import os
import shutil

from tqdm import tqdm

import numpy as np
import pandas as pd
import tensorflow as tf

import librosa

from conf import (
    DATA_PATH,
    COLUMNS,
    AUDIOPARAMS,
    STFTPARAMS,
    MELPARAMS,
    MFCCPAMARS,
    CACHEDIR,
    EXPERIMENTS
)


def get_csv_path(experiment):
    return os.path.join(DATA_PATH, f"{experiment}.csv")


def get_recording(experiment):
    return experiment.split('-')[-1].split('.csv')[0]


def get_nest_number(experiment):
    return experiment.split('N')[-1].split('E')[0]


def get_postnatalday(experiment):
    return experiment.split('EP')[-1].split('-')[0]


def get_audio_path(recording):
    return os.path.join(DATA_PATH, f"T0000{recording}.WAV")


def create_dataframe(experiment):
    df = pd.read_csv(
        get_csv_path(experiment),
        header=0,
        na_values=0,
        keep_default_na=False)[COLUMNS]
    df = df.assign(experiment=experiment)
    return df


def load_audio(path, offset, duration):
    return librosa.core.load(
        path=path.numpy(),
        offset=offset.numpy(),
        duration=duration.numpy(),
        **AUDIOPARAMS)


def compute_stft(audio):
    return np.abs(librosa.stft(y=audio.numpy(), **STFTPARAMS))


def compute_melspectrogram(spectrogram):
    return librosa.feature.melspectrogram(S=spectrogram, **MELPARAMS)


def compute_mfcc(melspectrogram):
    tmp = librosa.feature.mfcc(
        S=librosa.power_to_db(melspectrogram), **MFCCPAMARS)
    width = min(tmp.shape[1], 9)
    MFCC = tmp[1:, :]
    d_MFCC = librosa.feature.delta(tmp, width=width)
    dd_MFCC = librosa.feature.delta(d_MFCC, width=width)
    return np.vstack([MFCC, d_MFCC, dd_MFCC])


def load_audio_tf(sample):
    return tf.py_function(
        func=load_audio,
        inp=[sample['audio_path'], sample['t0'], sample['duration']],
        Tout=tf.float32)


def compute_stft_tf(sample):
    return tf.py_function(
        func=compute_stft,
        inp=[sample['audio']],
        Tout=tf.float32)


def compute_melspectrogram_tf(sample):
    return tf.py_function(
        func=compute_melspectrogram,
        inp=[sample['stft']],
        Tout=tf.float32)


def compute_mfcc_tf(sample):
    return tf.py_function(
        func=compute_mfcc,
        inp=[sample['mel']],
        Tout=tf.float32)


def load_audio_dataset(sample):
    return dict(sample, audio=load_audio_tf(sample))


def compute_stft_dataset(sample):
    return dict(sample, stft=compute_stft_tf(sample))


def compute_melspectrogram_dataset(sample):
    return dict(sample, mel=compute_melspectrogram_tf(sample))


def compute_mfcc_dataset(sample):
    return dict(sample, mfcc=compute_mfcc_tf(sample))


def check_tensor_shape(tensor):
    return tf.math.greater_equal(tf.shape(tensor)[1], 9)


def reduce_time_mean(tensor):
    return tf.math.reduce_mean(tensor, axis=1, keepdims=False)


def reduce_time_max(tensor):
    return tf.math.reduce_max(tensor, axis=1, keepdims=False)


def inverse(tensor):
    return tf.math.multiply(-1.0, tensor)


def get_dataframe():

    df = pd.concat([
        create_dataframe(e) for e in EXPERIMENTS
    ]).rename({"sample": "sample_nb"}, axis=1)

    df = df.assign(recording=df.experiment.apply(get_recording))
    df = df.assign(nest=df.experiment.apply(get_nest_number))
    df = df.assign(postnatalday=df.experiment.apply(get_postnatalday))
    df = df.assign(audio_path=df.recording.apply(get_audio_path))
    df = df.assign(t0=df.t0.apply(lambda f: float(f.replace(',', '.'))))
    df = df.assign(t1=df.t1.apply(lambda t: float(t.replace(',', '.'))))
    df = df.assign(
        duration=df['t1'].astype(np.float)-df['t0'].astype(np.float))

    # shuffle data frame
    df = df.sample(frac=1)

    return df


def get_dataset(recompute=False, element_spec=False):

    df = get_dataframe()

    dataset = (
        tf.data.Dataset.from_tensor_slices({
            key: df[key].values for key in df
            }
        )
    )

    # compute audio
    dataset = dataset.map(load_audio_dataset)

    # compute audio stft
    dataset = dataset.map(compute_stft_dataset)

    # filter out bad shape stft
    dataset = dataset.filter(lambda sample: check_tensor_shape(sample["stft"]))

    # compute audio features
    dataset = dataset.map(compute_melspectrogram_dataset)
    dataset = dataset.map(compute_mfcc_dataset)

    # map dynamic compression
    C = 1000
    dataset = dataset.map(
        lambda sample: dict(sample, stft=tf.math.log(1+C*sample["stft"])))
    dataset = dataset.map(
        lambda sample: dict(sample, mel=tf.math.log(1+C*sample["mel"])))

    # compute time aggregated features
    dataset = dataset.map(
        lambda sample: dict(
            sample, mean_stft=reduce_time_mean((sample['stft']))))
    dataset = dataset.map(
        lambda sample: dict(
            sample, mean_mel=reduce_time_mean((sample['mel']))))
    dataset = dataset.map(
        lambda sample: dict(
            sample, mean_mfcc=reduce_time_mean((sample['mfcc']))))

    dataset = dataset.map(
        lambda sample: dict(sample, inv_stft=inverse((sample['stft']))))
    dataset = dataset.map(
        lambda sample: dict(sample, inv_mel=inverse((sample['mel']))))
    dataset = dataset.map(
        lambda sample: dict(sample, inv_mfcc=inverse((sample['mfcc']))))

    dataset = dataset.map(
        lambda sample: dict(
            sample, max_stft=reduce_time_max((sample['inv_stft']))))
    dataset = dataset.map
    (lambda sample: dict(sample, max_mel=reduce_time_max((sample['inv_mel']))))
    dataset = dataset.map(
        lambda sample: dict(
            sample, max_mfcc=reduce_time_max((sample['inv_mfcc']))))

    dataset = dataset.cache(CACHEDIR)

    if element_spec:
        print(dataset.element_spec)

    if recompute:
        try:
            shutil.rmtree(CACHEDIR, ignore_errors=True)
            os.mkdir(CACHEDIR)
            mss = f"Directory {CACHEDIR} has been removed"
            mss += "and recreated successfully."
            print(mss)
        except Exception as e:
            print(e)

    iterator = iter(dataset)
    for sample in tqdm(iterator):
        pass

    return dataset
