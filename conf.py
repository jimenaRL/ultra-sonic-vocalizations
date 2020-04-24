import os
import librosa



BASE_PATH = "/home/utilisateur/Desktop/palomars"

# CACHEDIR = os.path.join(BASE_PATH, "dataset", "features")
CACHEDIR = os.path.join(BASE_PATH, "dataset", "features-nest")

# metadata parameters
# N1EP09--> 494; N1EP04-->482; N2EP01-->637
# N1= nest numero 1, E= experince; P09=postnatal day 09

EXPERIMENTS = [
    "N1EP04-482",
    "N1EP09-494",
    "N2EP01-637" 
]


DATA_PATH = os.path.join(BASE_PATH, 'data')
PLOTS_PATH = os.path.join(BASE_PATH, 'plots')

COLUMNS = ["sample", "t0", "t1", "vocalization", "file"]
VOCALIZATIONS = {
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

FFTFREQS = librosa.core.fft_frequencies(sr=AUDIOPARAMS['sr'], n_fft=STFTPARAMS['n_fft'])


# MEL FILTERBANK PARAMS
NMELS = 128
HTK = True

MELPARAMS = {
    'sr': SR, 'n_fft': NFFT, 'n_mels': NMELS, 'fmin': FMIN, 'fmax': FMAX, 'htk': HTK
}

# MFCC PARAMS
MFCCPAMARS = {
    'sr': SR,
    'n_mfcc': 13,
    'dct_type': 2,
    'norm': 'ortho',
    'htk': HTK
}

melfb =  librosa.filters.mel(**MELPARAMS)
print(f"mel fiterbank shape = {melfb.shape}")
print(f"STFT time resolution = {1000 * MELPARAMS['n_fft']/AUDIOPARAMS['sr']} ms")
print(f"STFT frequency resolution = {MELPARAMS['fmax']/MELPARAMS['n_fft']} Hz\n")