import tensorflow as tf
import numpy as np
import librosa
import librosa.feature as lf

from audiovocana.conf import (
    # AUDIOPARAMS,
    STFTPARAMS,
    MELPARAMS,
    MFCCPAMARS,
    ZEROCRPARAMS,
    SPECTRALFLATNESSPARAMS,
    SPECTRALCENTROIDPARAMS,
    SPECTRALBANDWIDTHPARAMS,
    MIN_STFT_LENGTH
)

import audiovocana.ffmpeg_utils.ffmpeg_utils as ffmpeg


def load_audio(path, offset, duration):
    try:
        waveform, sr = ffmpeg.load_audio_file(
            audio_filename=path.numpy(),
            start_second=offset.numpy(),
            duration_second=duration.numpy())
        # waveform, sr = librosa.core.load(
        #             path=path.numpy(),
        #             offset=offset.numpy(),
        #             duration=duration.numpy(),
        #             **AUDIOPARAMS)
        return waveform.flatten(), False
    except Exception as e:
        print(f"load_audio error: {e}")
        return np.array([-1.0]), True


def compute_zero_crossing_rate(audio):
    try:
        return lf.zero_crossing_rate(
            y=audio.numpy(), **ZEROCRPARAMS).flatten(), False
    except Exception as e:
        print(f"compute_zero_crossing_rate error: {e}")
        return np.array([-1.0]), True


def compute_spectral_flatness(spectrogram):
    try:
        return lf.spectral_flatness(
            S=spectrogram, **SPECTRALFLATNESSPARAMS).flatten(), False
    except Exception as e:
        print(f"compute_spectral_flatness error: {e}")
        return np.array([-1.0]), True


def compute_spectral_bandwidth(audio):
    try:
        return lf.spectral_bandwidth(
            y=audio.numpy(), **SPECTRALBANDWIDTHPARAMS).flatten(), False
    except Exception as e:
        print(f"compute_spectral_bandwidth error: {e}")
        return np.array([-1.0]), True


def compute_spectral_centroid(spectrogram):
    try:
        return lf.spectral_centroid(
            S=spectrogram, **SPECTRALCENTROIDPARAMS).flatten(), False
    except Exception as e:
        print(f"compute_spectral_centroid error: {e}")
        return np.array([-1.0]), True


def compute_stft(audio):
    try:
        return np.abs(librosa.stft(y=audio.numpy(), **STFTPARAMS)), False
    except Exception as e:
        print(f"compute_stft error: {e}")
        return np.array([-1.0, -1.0]), True


def compute_melspectrogram(spectrogram):
    try:
        return lf.melspectrogram(
            S=spectrogram, **MELPARAMS), False
    except Exception as e:
        print(f"compute_melspectrogram error: {e}")
        return np.array([-1.0, -1.0]), True


def compute_mfcc(melspectrogram):
    try:
        tmp = lf.mfcc(
            S=librosa.power_to_db(melspectrogram), **MFCCPAMARS)
        MFCC = tmp[1:, :]
        d_MFCC = lf.delta(tmp, width=MIN_STFT_LENGTH)
        dd_MFCC = lf.delta(d_MFCC, width=MIN_STFT_LENGTH)
        return np.vstack([MFCC, d_MFCC, dd_MFCC]), False
    except Exception as e:
        print(f"compute_mfcc error: {e}")
        return np.array([-1.0, -1.0]), True


def load_audio_tf(sample):
    res = tf.py_function(
        func=load_audio,
        inp=[sample['audio_path'], sample['t0'], sample['duration']],
        Tout=(tf.float32, tf.bool))
    return dict(list(sample.items()) + [("audio", res[0]), ("error", res[1])])


def compute_zero_crossing_rate_tf(sample):
    res = tf.py_function(
        func=compute_zero_crossing_rate,
        inp=[sample['audio']],
        Tout=(tf.float32, tf.bool))
    new_items = [("zcr", res[0]), ("error", res[1])]
    return dict(list(sample.items()) + new_items)


def compute_spectral_flatness_tf(sample):
    res = tf.py_function(
        func=compute_spectral_flatness,
        inp=[sample['stft']],
        Tout=(tf.float32, tf.bool))
    new_items = [("sf", res[0]), ("error", res[1])]
    return dict(list(sample.items()) + new_items)


def compute_spectral_bandwidth_tf(sample):
    res = tf.py_function(
        func=compute_spectral_bandwidth,
        inp=[sample['audio']],
        Tout=(tf.float32, tf.bool))
    new_items = [("sbw", res[0]), ("error", res[1])]
    return dict(list(sample.items()) + new_items)


def compute_spectral_centroid_tf(sample):
    res = tf.py_function(
        func=compute_spectral_centroid,
        inp=[sample['stft']],
        Tout=(tf.float32, tf.bool))
    new_items = [("sc", res[0]), ("error", res[1])]
    return dict(list(sample.items()) + new_items)


def compute_stft_tf(sample):
    res = tf.py_function(
        func=compute_stft,
        inp=[sample['audio']],
        Tout=(tf.float32, tf.bool))
    return dict(list(sample.items()) + [("stft", res[0]), ("error", res[1])])


def compute_melspectrogram_tf(sample):
    res = tf.py_function(
        func=compute_melspectrogram,
        inp=[sample['stft']],
        Tout=(tf.float32, tf.bool))
    return dict(list(sample.items()) + [("mel", res[0]), ("error", res[1])])


def compute_mfcc_tf(sample):
    res = tf.py_function(
        func=compute_mfcc,
        inp=[sample['mel']],
        Tout=(tf.float32, tf.bool))
    return dict(list(sample.items()) + [("mfcc", res[0]), ("error", res[1])])
