import os
import librosa
from glob import glob
import numpy as np

SEED = 666

FFMPEG_BINARY = os.environ["AUDIOVOCANA_FFMPEG_BINARY"]
FFPROBE_BINARY = os.environ["AUDIOVOCANA_FFPROBE_BINARY"]

BASE_PATH = os.environ["AUDIOVOCANA_BASE_PATH"]
XLSX_FOLDER = os.environ["AUDIOVOCANA_XLSX_FOLDER"]
AUDIO_FOLDER = os.environ["AUDIOVOCANA_AUDIO_FOLDER"]

print("~~~~~~ AUDIOVOCANA PARAMS ~~~~~~")
print(f"BASE_PATH = {BASE_PATH}")
print(f"XLSX_FOLDER = {XLSX_FOLDER}")
print(f"AUDIO_FOLDER = {AUDIO_FOLDER}")


CACHEDIR = os.path.join(BASE_PATH, "cache")
PLOTS_PATH = os.path.join(BASE_PATH, 'plots')


XLSX_FILES = glob(os.path.join(XLSX_FOLDER, "*.xlsx"))


PRECOLUMNS = [
    "event", "C", "D", "E", "F", "t0", "t1",
    "I", "J", "K", "L", "M", "N", "vocalization"
]
COLUMNS = {
    "t0": np.float,
    "t1": np.float,
    "duration": np.float,
    "event": np.int,
    "postnatalday": np.int,
    "vocalization": np.int,
    "nest": str,
    "year": str,
    "audio_path": str,
    "experiment": str,
    "recording": str
}

MISSING_VOCALIZATION_LABEL = 0

VOCALIZATIONS = {
    0: "MISSING",
    1: "audible",
    2: "USV"
}

COLORS = {
    1: 'tab:blue',
    2: 'tab:orange'
}

# AUDIO PARAMS
SR = int(250e3)
FMIN = 0
FMAX = 0.5 * SR

AUDIOPARAMS = {
    'sr': SR,
    'mono': True,
}

# STFT PARAMS
NFFT = 2048*4
HOPLENGTH = int(NFFT/16)
WINLENGTH = NFFT

STFTPARAMS = {
    'n_fft': NFFT,
    'hop_length': HOPLENGTH,
    'win_length': WINLENGTH
}

FFTFREQS = librosa.core.fft_frequencies(
    sr=AUDIOPARAMS['sr'],
    n_fft=STFTPARAMS['n_fft']
)


# MEL FILTERBANK PARAMS
NMELS = 128
HTK = True

MELPARAMS = {
    'sr': SR,
    'n_fft': NFFT,
    'n_mels': NMELS,
    'fmin': FMIN,
    'fmax': FMAX,
    'htk': HTK
}

# MFCC PARAMS
MFCCPAMARS = {
    'sr': SR,
    'n_mfcc': 13,
    'dct_type': 2,
    'norm': 'ortho',
    'htk': HTK
}

MELFB = librosa.filters.mel(**MELPARAMS)

print(f"mel fiterbank shape = {MELFB.shape}")
print(
    f"STFT time resolution = {1000 * MELPARAMS['n_fft']/AUDIOPARAMS['sr']} ms")
print(
    f"STFT frequency resolution = {MELPARAMS['fmax']/MELPARAMS['n_fft']} Hz\n")


# DATASET PARAMS
MIN_WAVEFORM_LENGTH = 3
MIN_STFT_LENGTH = 9
