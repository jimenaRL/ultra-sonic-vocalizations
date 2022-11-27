import numpy as np
import scipy as sp

import matplotlib.pyplot as plt

import librosa
from librosa.core import frames_to_time

from audiovocana.conf import SR, FMAX
import audiovocana.ffmpeg_utils.ffmpeg_utils as ffmpeg

# FFT PARAMS FOR f0 ESTIMATION FROM SPECTROGRAMS
HOP_LENGTH = 16 #8
WIN_LENGTH = 400 # 200
NFFT = WIN_LENGTH
NBFFTBINS = 1 + NFFT / 2

print(f"f0 estimation STFT frequency resolution = {FMAX / NBFFTBINS} Hz.")
print(f"STFT time resolution = {1e3 * NFFT / SR} ms.")


FFTPLOTPARAMS = {
    # for x axis settings
    'sr': SR,
    'hop_length': HOP_LENGTH,
    'x_axis': 'time',
    # for y axis settings
    'fmax': FMAX,
    'y_axis': 'linear',
    # color
    #'cmap': 'gray', # 'gray', 'PuBu_r', 'RdBu', 'bone'
    'vmin': -80,  # -50, -25
    'vmax': 0
}

FFTPARAMS = {
    'n_fft': NFFT,
    'hop_length': HOP_LENGTH,
    'win_length': WIN_LENGTH,
    'center': True,
    'pad_mode': 'reflect',
    'window': 'hann'
}


def find_outliers(data, n_quintiles):
    data = np.abs(np.diff(data))
    sort_data = np.sort(data)

    qs = [i*100/n_quintiles for i in range(n_quintiles)]
    qs = list(map(lambda l: np.percentile(data, l, interpolation = 'midpoint'),  qs))
    
    IQR = qs[-1] - qs[0] 
    low_lim = qs[0] - 1.5 * IQR / FMAX
    up_lim = qs[-1] + 1.5 * IQR

    A = low_lim<=data
    B = data<=up_lim 

    res = (A * B)
    return res

def mean_2dn_derivative(x, y):
    spl = sp.interpolate.splrep(x, y, k=3) # no smoothing, 3rd order spline
    ddy = sp.interpolate.splev(x, spl, der=2) # use those knots to get second derivative 
    return np.mean(np.abs(ddy))

def f0_estimation(
    row,
    tonality_threshold=0.85,
    spline_degree=3,
    title='',
    image_path=None,
    show=False):

    # (0)
    t = row.t0
    
    y, sr = ffmpeg.load_audio_file(
        audio_filename=row.audio_path,
        start_second=row.t0,
        duration_second=row.duration)

    y = y.flatten()
        
    # amplitud spectrogram
    D = np.abs(librosa.stft(y, **FFTPARAMS))

    # compute time coordinates
    x_lim = frames_to_time(np.arange(D.shape[1] + 1), sr=SR, hop_length=HOP_LENGTH)[-1]
    x = np.linspace(0, x_lim, D.shape[1])
    
    # (1) plot STFT
    if show:
        fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(16, 8))
        img = librosa.display.specshow(
            librosa.amplitude_to_db(D, ref=np.max), ax=ax[0], **FFTPLOTPARAMS)
        _title = f'time = {t} \n {title}'
        ax[0].set(title=_title)

    # (2) compute contour = max f0 over STFT
    
    # Spectrogram filtering by nearest-neighbors (image denoising)
    rec = librosa.segment.recurrence_matrix(
        D, mode='affinity', metric='cosine', sparse=True)
    D = librosa.decompose.nn_filter(
        D, rec=rec, aggregate=np.average)

    if show:
        img = librosa.display.specshow(
            librosa.amplitude_to_db(D, ref=np.max), ax=ax[1], **FFTPLOTPARAMS)

    # ignore bass frequencies on powerspectrogram
    f_cut = 16
    S = D[f_cut:, :]**2 
    freqs = np.linspace(0, FMAX, D.shape[0])[f_cut:]

    # prepare array for peak max f0 at each time bin
    f0_max = np.zeros(S.shape[1])

    A = np.argmax(S, axis=0)
    for t, f in enumerate(A):
        f0_max[t] = freqs[f]

    if show:
        ax[1].plot(x, f0_max, color='r', marker='^', linewidth=0, label='max')
        ax[1].legend(loc='upper right')

    # (3) tonality thresholding of contour
    # A high spectral flatness (closer to 1.0) indicates
    # the spectrum is similar to white noise
    flateness = librosa.feature.spectral_flatness(S=S, power=1.0).flatten()
    f0_max_th = f0_max[flateness < tonality_threshold]
    
    x1 = x[flateness < tonality_threshold]

    if show:
        ax[1].plot(x1, f0_max_th, color='g', marker='.', linewidth=0, label='max_th')
        ax[1].legend(loc='upper right')

    # (4) outliers removing from contour
    outliers = find_outliers(np.abs(np.diff(f0_max_th)), n_quintiles=20)
    f0_max_th_out = f0_max_th[2:][outliers]

    x2 = x1[2:][outliers]

    # (4) contour smoothing by spline interpolation
    tck = sp.interpolate.splrep(x2, f0_max_th_out/FMAX, k=3, s=0.01)
    f0_smooth = FMAX * sp.interpolate.splev(x, tck, der=0)
    mf0 = int(np.mean(f0_smooth))

    # (5) compute mean normalize 2nd derivative from normalized contou
    ddy = sp.interpolate.splev(x, tck, der=2)
    mnddy = np.mean(np.abs(ddy))

    if show:
        ax[1].plot(x2, f0_max_th_out, color='b', marker='x', linewidth=0, label='max_th_out')
        ax[1].legend(loc='upper right')

        ax[1].plot(x, f0_smooth, color='m', marker='x', linewidth=0, label='f0_smooth')
        ax[1].legend(loc='upper right')

        _title = ''
        #_title += 'Spectrogram and estimated f0'
        _title += f'\n mean normalized 2dn derivative = {mnddy* 1e-3}'
        _title += f'\n mean f0 = {mf0}'
        plt.title(_title)
        plt.tight_layout()
        
        fig.colorbar(img, ax=ax, format="%+2.f dB")

        #save image
        if image_path:
            fig.savefig(image_path)        

    return mf0, mnddy 