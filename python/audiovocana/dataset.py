import os
import shutil

from tqdm import tqdm

import tensorflow as tf

from audiovocana.preprocessing import get_dataframe

from audiovocana.conf import (
    MIN_WAVEFORM_LENGTH,
    SEED
)
from audiovocana.audio_features import (
    load_audio_tf,
    compute_stft_tf,
    compute_mfcc_tf,
    compute_melspectrogram_tf,
    compute_spectral_centroid_tf,
    compute_spectral_bandwidth_tf,
    compute_spectral_flatness_tf,
    compute_zero_crossing_rate_tf
)


def manage_cache(dataset, cache_folder, recompute):
    if cache_folder:
        os.makedirs(cache_folder, exist_ok=True)
        cache_prefix = os.path.join(cache_folder, "dataset")
        dataset = dataset.cache(cache_prefix)
        if recompute:
            try:
                shutil.rmtree(cache_folder, ignore_errors=True)
                os.mkdir(cache_folder)
                mss = f"Directory {cache_folder} has been removed"
                mss += "and recreated successfully."
                print(mss)
            except Exception as e:
                print(e)
    return dataset


def check_tensor_shape(tensor, d, minlength):
    return tf.math.greater_equal(tf.shape(tensor)[d], minlength)


def reduce_time_mean(tensor, axis=1):
    return tf.math.reduce_mean(tensor, axis=axis, keepdims=False)


def reduce_time_max(tensor):
    return tf.math.reduce_max(tensor, axis=1, keepdims=False)


def inverse(tensor):
    return tf.math.multiply(-1.0, tensor)


def get_dataset(
    csv_path=None,
    shuffle=True,
    cache_folder=None,
    recompute=False
):

    df = get_dataframe(csv_path=csv_path, save=False)

    # create dataset
    dataset = (
        tf.data.Dataset.from_tensor_slices({
            key: df[key].values for key in df
            }
        )
    )

    # compute audio
    dataset = dataset.map(load_audio_tf)

    # filter out bad shaped audio
    dataset = dataset.filter(
        lambda sample: check_tensor_shape(
            sample["audio"], 0, MIN_WAVEFORM_LENGTH))

    # compute audio stft
    dataset = dataset.map(compute_stft_tf)

    # compute audio features
    dataset = dataset.map(compute_melspectrogram_tf)
    dataset = dataset.map(compute_mfcc_tf)
    dataset = dataset.map(compute_spectral_centroid_tf)
    dataset = dataset.map(compute_spectral_bandwidth_tf)
    dataset = dataset.map(compute_spectral_flatness_tf)
    dataset = dataset.map(compute_zero_crossing_rate_tf)

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
        lambda sample: dict(
            sample, mean_zrc=reduce_time_mean((sample['zcr']), axis=0)))
    dataset = dataset.map(
        lambda sample: dict(
            sample, mean_sbw=reduce_time_mean((sample['sbw']), axis=0)))
    dataset = dataset.map(
        lambda sample: dict(
            sample, mean_sf=reduce_time_mean((sample['sf']), axis=0)))
    dataset = dataset.map(
        lambda sample: dict(
            sample, mean_sc=reduce_time_mean((sample['sc']), axis=0)))

    dataset = dataset.map(
        lambda sample: dict(sample, inv_stft=inverse((sample['stft']))))
    dataset = dataset.map(
        lambda sample: dict(sample, inv_mel=inverse((sample['mel']))))
    dataset = dataset.map(
        lambda sample: dict(sample, inv_mfcc=inverse((sample['mfcc']))))

    dataset = dataset.map(
        lambda sample: dict(
            sample, max_stft=reduce_time_max((sample['inv_stft']))))
    dataset = dataset.map(
        lambda sample: dict(
            sample, max_mel=reduce_time_max((sample['inv_mel']))))
    dataset = dataset.map(
        lambda sample: dict(
            sample, max_mfcc=reduce_time_max((sample['inv_mfcc']))))

    # shuffle
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10, seed=SEED)

    dataset = manage_cache(dataset, cache_folder, recompute)

    if recompute:
        for sample in tqdm(iter(dataset)):
            pass

    return dataset
