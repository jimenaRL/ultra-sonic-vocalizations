import numpy as np
import scipy as sp

import matplotlib.pyplot as plt

import librosa
from librosa import display
from librosa.core import frames_to_time

from audiovocana.conf import SR, FMAX
import audiovocana.ffmpeg_utils.ffmpeg_utils as ffmpeg

# FFT PARAMS FOR f0 ESTIMATION FROM SPECTROGRAMS
from audiovocana.conf import (
    HOPLENGTH,
    WINLENGTH,
    NFFT,
    NBFFTBINS,
    STFTPARAMS
)
print(f"f0 estimation STFT frequency resolution = {FMAX / NBFFTBINS} Hz.")
print(f"STFT time resolution = {1e3 * NFFT / SR} ms.")

STFTPLOTPARAMS = {
    # for x axis settings
    'sr': SR,
    'hop_length': HOPLENGTH,
    'x_axis': 'time',
    # for y axis settings
    'fmax': FMAX,
    'y_axis': 'linear',
    # color
    # 'cmap': 'gray', # 'gray', 'PuBu_r', 'RdBu', 'bone'
    'vmin': -80,  # -50, -25
    'vmax': 0
}


def find_outliers(data, n_quintiles):
    data = np.abs(np.diff(data))

    qs = [i * 100 / n_quintiles for i in range(n_quintiles)]
    qs = list(map(
        lambda l: np.percentile(data, l, interpolation='midpoint'), qs))

    IQR = qs[-1] - qs[0]
    low_lim = qs[0] - 1.5 * IQR / FMAX
    up_lim = qs[-1] + 1.5 * IQR

    A = low_lim<=data
    B = data<=up_lim

    res = (A * B)
    return res


def mean_2dn_derivative(x, y):
    # no smoothing, 3rd order spline
    spl = sp.interpolate.splrep(x, y, k=3)
    # use those knots to get second derivative
    ddy = sp.interpolate.splev(x, spl, der=2)
    return np.mean(np.abs(ddy))


def f0_estimation(
    audio_path,
    t0,
    duration,
    tonality_threshold=0.85,
    smoothness=0.01,
    ignore_border=0.15,
    title='',
    image_path=None,
    show=False):


    y, sr = ffmpeg.load_audio_file(
        audio_filename=audio_path,
        start_second=t0,
        duration_second=duration)

    y = y.flatten()

    # amplitud spectrogram
    D = np.abs(librosa.stft(y, **STFTPARAMS))

    # compute time coordinates
    x_lim = frames_to_time(
        np.arange(D.shape[1] + 1), sr=SR, hop_length=HOPLENGTH)[-1]
    x = np.linspace(0, x_lim, D.shape[1])

    # (1) plot STFT
    if show:
        fig, ax = plt.subplots(
            nrows=2, sharex=False, sharey=False, figsize=(16, 14))
        img = librosa.display.specshow(
            librosa.amplitude_to_db(D, ref=np.max), ax=ax[1], **STFTPLOTPARAMS)
        _title = f'duration = {10**3 * duration:3.0f} ms \n {title}'
        ax[0].set(title=_title)

    # (2) compute contour = max f0 over STFT

    # Spectrogram filtering by nearest-neighbors (image denoising)
    rec = librosa.segment.recurrence_matrix(
        D, mode='affinity', metric='cosine', sparse=True)
    D = librosa.decompose.nn_filter(D, rec=rec, aggregate=np.average)

    if show:
        img = librosa.display.specshow(
            librosa.amplitude_to_db(D, ref=np.max), ax=ax[0], **STFTPLOTPARAMS)

    # ignore bass frequencies on powerspectrogram
    f_cut = 16
    S = D[f_cut:, :]
    Spower = S**2
    freqs = np.linspace(0, FMAX, D.shape[0])[f_cut:]

    # prepare array for peak max f0 at each time bin
    f0 = np.zeros(S.shape[1])

    A = np.argmax(S, axis=0)
    for t, f in enumerate(A):
        f0[t] = freqs[f]

    if show:
        ax[0].plot(x, f0, color='r', marker='^', linewidth=0, label='max')
        ax[0].legend(loc='upper right')

    # (3) tonality thresholding of contour
    # A high spectral flatness (closer to 1.0) indicates
    # the spectrum is similar to white noise
    flateness = librosa.feature.spectral_flatness(
        S=Spower, power=1.0).flatten()
    f0_tonality = f0[flateness < tonality_threshold]

    x1 = x[flateness < tonality_threshold]
    S1 = S[:, flateness < tonality_threshold]

    if show:
        ax[0].plot(
            x1, f0_tonality, color='g', marker='.', linewidth=0, label='f0_tonality')
        ax[0].legend(loc='upper right')

    # (4) outliers removing from contour
    outliers = find_outliers(np.abs(np.diff(f0_tonality)), n_quintiles=20)
    f0_tonality_outliers = f0_tonality[2:][outliers]

    x2 = x1[2:][outliers]
    S2 = S1[:, 2:][:, outliers]

    # (4) contour smoothing by spline interpolation
    tck = sp.interpolate.splrep(x2, f0_tonality_outliers / FMAX, k=3, s=smoothness)
    f0_smooth = FMAX * sp.interpolate.splev(x, tck, der=0)
    # take only central part as borders are noisy
    t = ignore_border
    k = len(f0_smooth)
    start = int(t * k)
    stop = int((1 - t) * k)
    mean_f0 = int(np.mean(f0_smooth[start:stop]))
    max_f0 = int(np.max(f0_smooth[start:stop]))
    min_f0 = int(np.min(f0_smooth[start:stop]))

    # (5) compute mean normalize 2nd derivative from normalized contou
    ddy = sp.interpolate.splev(x, tck, der=2)
    # take only central part as borders are noisy
    t = ignore_border
    m = len(ddy)
    no_border_ddy = ddy[int(t * m):int((1 - t) * m)]
    mnddy = np.mean(np.abs(no_border_ddy))

    rms = librosa.feature.rms(S=S2, frame_length=168, hop_length=42).flatten()
    mean_rms = np.mean(rms)

    if show:
        ax[0].plot(
            x2, f0_tonality_outliers,
            color='b', marker='x', linewidth=0, label='f0_tonality_outliers')
        ax[0].legend(loc='upper right')

        ax[0].plot(
            x, f0_smooth,
            color='m', marker='x', linewidth=0, label='f0_smooth')
        ax[0].legend(loc='upper right')

        _title = ''
        _title += f'\n mean normalized 2dn derivative = {mnddy* 1e-3}'
        _title += f'\n mean rms = {mean_rms}'
        _title += f'\n max f0 = {max_f0}'
        _title += f'\n mean f0 = {mean_f0}'
        _title += f'\n min f0 = {min_f0}'
        plt.title(_title)
        plt.tight_layout()

        fig.colorbar(img, ax=ax, format="%+2.f dB")

        if image_path:
            fig.savefig(image_path)

    return {
        'min_f0': min_f0,
        'max_f0': max_f0,
        'mean_f0': mean_f0,
        'mean_2dn_derivative': mnddy,
        'mean_rms': mean_rms
    }
