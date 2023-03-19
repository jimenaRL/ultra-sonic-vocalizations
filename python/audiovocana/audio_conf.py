import os
import librosa

SEED = 666

try:
    FFMPEG_BINARY = os.environ["FFMPEG_BINARY"]
except Exception as e:
    print(f"Missing environ variable 'FFMPEG_BINARY'.")
try:
    FFPROBE_BINARY = os.environ["FFPROBE_BINARY"]
except Exception as e:
    print(f"Missing environ variable 'FFPROBE_BINARY'.")

# AUDIO PARAMS
SR = int(250e3)
FMIN = 0
FMAX = 0.5 * SR

AUDIOPARAMS = {
    'sr': SR,
    'mono': True,
}

# --------------------------------------------------------------------

# 1st  set paramaters, used in results from April 10th, 2020
# NFFT = 2048*4
# HOPLENGTH = int(NFFT/16)
# CENTER_WINDOWS = True
# NMELS = 128
# MIN_STFT_LENGTH = 9

# 2nd  set paramaters, used by Paloma, more close to avisoft software ones
NFFT = 200
HOPLENGTH = 8
CENTER_WINDOWS = True
NMELS = 32
MIN_STFT_LENGTH = 9


# --------------------------------------------------------------------

# STFT PARAMS
NFFT = NFFT
HOPLENGTH = HOPLENGTH
WINLENGTH = NFFT
NBFFTBINS = 1 + NFFT / 2
CENTER_WINDOWS = CENTER_WINDOWS
WINDOW = 'hann'

STFTPARAMS = {
    'n_fft': NFFT,
    'hop_length': HOPLENGTH,
    'win_length': WINLENGTH,
    'center': CENTER_WINDOWS,
    'window': WINDOW,
    # 'pad_mode': 'reflect',
}


# FFTFREQS = librosa.core.fft_frequencies(
#     sr=AUDIOPARAMS['sr'],
#     n_fft=STFTPARAMS['n_fft']
# )


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

# MELFREQS = librosa.core.mel_frequencies(
#     n_mels=NMELS,
#     fmin=FMIN,
#     fmax=FMAX,
#     htk=HTK)
# MELFB = librosa.filters.mel(**MELPARAMS)


# OTHERS PARAMS

SPECTRALCENTROIDPARAMS = {
    'sr': SR,
    'n_fft': NFFT,
    'hop_length': HOPLENGTH,
    'freq': None,
}


SPECTRALBANDWIDTHPARAMS = {
    'sr': SR,
    'n_fft': NFFT,
    'hop_length': HOPLENGTH,
    'freq': None,
    'centroid': None,
    'norm': True,
    'p': 2
}


SPECTRALFLATNESSPARAMS = {
    'n_fft': NFFT,
    'hop_length': HOPLENGTH,
    'amin': 1e-10,
    'power': 2.0
}

ZEROCRPARAMS = {
    'frame_length': 2048,
    'hop_length': 512,
    'center': CENTER_WINDOWS,
}


# F0 PARAMS
F0PARAMS = {
    # (3) tonality thresholding of contour
    # A high spectral flatness (closer to 1.0) indicates
    # the spectrum is similar to white noise
    #     f0_max_th = f0_max[flateness < tonality_threshold]
    'tonality_threshold': 0.75,
    # (4) contour smoothing by spline interpolation
    'smoothness': 0.5,
    # take only central part as borders are noisy
    'ignore_border': 0.15
}

# PLOT STFT PARAMS

STFTPLOTPARAMS = {
    # for x axis settings
    'sr': SR,
    'hop_length': HOPLENGTH,
    'x_axis': 'ms',
    # for y axis settings
    'fmax': FMAX,
    'y_axis': 'linear',
    'cmap': 'magma'  # 'gray', 'bone', 'magma'
}


# SET MIN AUDIO LENGTH ALLOWED
if CENTER_WINDOWS:
    MIN_WAVEFORM_LENGTH = int(WINLENGTH / 2) + (MIN_STFT_LENGTH - 1)
else:
    MIN_WAVEFORM_LENGTH = WINLENGTH + (MIN_STFT_LENGTH - 1) * HOPLENGTH
MIN_AUDIO_LENGHT_MS = MIN_WAVEFORM_LENGTH * 1e3 / SR


print("~~~~~~ AUDIOVOCANA SETTINGS ~~~~~~")
print(f"AUDIOPARAMS \n {AUDIOPARAMS}")
print(f"STFTPARAMS \n {STFTPARAMS}")
print(f"F0PARAMS \n {F0PARAMS}")
# print(f"SPECTRALCENTROIDPARAMS \n {SPECTRALCENTROIDPARAMS}")
# print(f"SPECTRALBANDWIDTHPARAMS \n {SPECTRALBANDWIDTHPARAMS}")
# print(f"SPECTRALFLATNESSPARAMS \n {SPECTRALFLATNESSPARAMS}")
# print(f"ZEROCRPARAMS \n {ZEROCRPARAMS}")
# print(f"MELPARAMS \n {MELPARAMS}")
# print(f"MFCCPAMARS \n {MFCCPAMARS}")
print(f"mel fiterbank shape = {MELFB.shape}")
print(f"Minimun waveform length accepted is {MIN_WAVEFORM_LENGTH} PCM points.")
print(f"Minimun audio duration accepted is {MIN_AUDIO_LENGHT_MS} miliseconds.")
print(
    f"STFT time resolution = {1e3 * NFFT / SR} ms.")
print(
    f"STFT frequency resolution = {FMAX / NBFFTBINS} Hz.")
