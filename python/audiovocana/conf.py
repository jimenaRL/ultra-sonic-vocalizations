import os
import librosa
import numpy as np

SEED = 666

try:
    FFMPEG_BINARY = os.environ["FFMPEG_BINARY"]
except Exception as e:
    print(f"Missing environ variable 'FFMPEG_BINARY'.")
try:
    FFPROBE_BINARY = os.environ["FFPROBE_BINARY"]
except Exception as e:
    print(f"Missing environ variable 'FFPROBE_BINARY'.")

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

# --------------------------------------------------------------------

# 1st paramaters set used in results from April 10th, 2020
NFFT = 2048*4
HOPLENGTH = int(NFFT/16)
CENTER_WINDOWS = True
NMELS = 128
MIN_STFT_LENGTH = 5


# 2nd paramaters set
# NFFT = 2048
# HOPLENGTH = int(NFFT/4)
# CENTER_WINDOWS = False
# NMELS = 64
# MIN_STFT_LENGTH = 9

# --------------------------------------------------------------------


# STFT PARAMS
NFFT = NFFT
HOPLENGTH = HOPLENGTH
WINLENGTH = NFFT
NBFFTBINS = 1 + NFFT / 2
CENTER_WINDOWS = CENTER_WINDOWS

STFTPARAMS = {
    'n_fft': NFFT,
    'hop_length': HOPLENGTH,
    'win_length': WINLENGTH,
    'center': CENTER_WINDOWS
}

FFTFREQS = librosa.core.fft_frequencies(
    sr=AUDIOPARAMS['sr'],
    n_fft=STFTPARAMS['n_fft']
)


# MEL FILTERBANK PARAMS
NMELS = NMELS
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

# SET MIN AUDIO LENGTH ALLOWED
MIN_STFT_LENGTH = MIN_STFT_LENGTH
MIN_WAVEFORM_LENGTH = WINLENGTH + (MIN_STFT_LENGTH - 1) * HOPLENGTH
MIN_AUDIO_LENGHT_MS = MIN_WAVEFORM_LENGTH * 1e3 / SR


print("~~~~~~ AUDIOVOCANA SETTINGS ~~~~~~")
print(f"AUDIOPARAMS \n {AUDIOPARAMS}")
print(f"STFTPARAMS \n {STFTPARAMS}")
print(f"MELPARAMS \n {MELPARAMS}")
print(f"MFCCPAMARS \n {MFCCPAMARS}")
print(f"mel fiterbank shape = {MELFB.shape}")
print(f"Minimun waveform length accepted is {MIN_WAVEFORM_LENGTH} PCM points.")
print(f"Minimun audio duration accepted is {MIN_AUDIO_LENGHT_MS} miliseconds.")
print(
    f"STFT time resolution = {1e3 * MELPARAMS['n_fft']/AUDIOPARAMS['sr']} ms.")
print(
    f"STFT frequency resolution = {MELPARAMS['fmax']/NBFFTBINS} Hz.")
