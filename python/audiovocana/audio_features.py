import tensorflow as tf
import numpy as np
import librosa

from audiovocana.conf import (
    AUDIOPARAMS,
    STFTPARAMS,
    MELPARAMS,
    MFCCPAMARS,
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


def compute_stft(audio):
    try:
        return np.abs(librosa.stft(y=audio.numpy(), **STFTPARAMS)), False
    except Exception as e:
        print(f"compute_stft error: {e}")
        return np.array([-1.0, -1.0]), True


def compute_melspectrogram(spectrogram):
    try:
        return librosa.feature.melspectrogram(
            S=spectrogram, **MELPARAMS), False
    except Exception as e:
        print(f"compute_melspectrogram error: {e}")
        return np.array([-1.0, -1.0]), True


def compute_mfcc(melspectrogram):
    try:
        tmp = librosa.feature.mfcc(
            S=librosa.power_to_db(melspectrogram), **MFCCPAMARS)
        width = min(tmp.shape[1], 9)
        MFCC = tmp[1:, :]
        d_MFCC = librosa.feature.delta(tmp, width=width)
        dd_MFCC = librosa.feature.delta(d_MFCC, width=width)
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
