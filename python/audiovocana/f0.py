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
    STFTPARAMS,
    STFTPLOTPARAMS
)
print(f"f0 estimation STFT frequency resolution = {FMAX / NBFFTBINS} Hz.")
print(f"STFT time resolution = {1e3 * NFFT / SR} ms.")


def find_outliers(data, n_quintiles):
    data = np.abs(np.diff(data))

    qs = [i * 100 / n_quintiles for i in range(n_quintiles)]
    qs = list(map(
        lambda l: np.percentile(data, l, interpolation='midpoint'), qs))

    IQR = qs[-1] - qs[0]
    low_lim = qs[0] - 1.5 * IQR / FMAX
    up_lim = qs[-1] + 1.5 * IQR

    A = low_lim <= data
    B = data <= up_lim

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
    tonality_threshold,
    smoothness,
    ignore_border):

    y, sr = ffmpeg.load_audio_file(
        audio_filename=audio_path,
        start_second=t0,
        duration_second=duration)

    y = y.flatten()

    S = librosa.stft(y, **STFTPARAMS)

    return f0_estimation_from_spect(
        S,
        tonality_threshold,
        smoothness,
        ignore_border
    )


def f0_estimation_from_spect(
    S,
    tonality_threshold,
    smoothness,
    ignore_border,
    plot=False,
    title=None,
    image_path=None
    ):

    spectrogram = S

    # (1) amplitud spectrogram
    D = np.abs(S)

    # compute time coordinates
    x_lim = frames_to_time(
        np.arange(D.shape[1] + 1), sr=SR, hop_length=HOPLENGTH)[-1]
    x = np.linspace(0, x_lim, D.shape[1])

    # (2) compute contour = max f0 over STFT

    # Spectrogram filtering by nearest-neighbors (image denoising)
    rec = librosa.segment.recurrence_matrix(
        D, mode='affinity', metric='cosine', sparse=True)
    D = librosa.decompose.nn_filter(D, rec=rec, aggregate=np.average)

    # ignore bass frequencies on powerspectrogram
    f_cut = 16
    S_cut = D[f_cut:, :]
    Spower = S_cut**2
    freqs = np.linspace(0, FMAX, D.shape[0])[f_cut:]

    # prepare array for peak max f0 at each time bin
    f0 = np.zeros(S_cut.shape[1])

    A = np.argmax(S_cut, axis=0)
    for t, f in enumerate(A):
        f0[t] = freqs[f]

    # (3) tonality thresholding of contour
    # A high spectral flatness (closer to 1.0) indicates
    # the spectrum is similar to white noise
    flateness = librosa.feature.spectral_flatness(
        S=Spower, power=1.0).flatten()
    f0_tonality = f0[flateness < tonality_threshold]

    x1 = x[flateness < tonality_threshold]
    S1 = S_cut[:, flateness < tonality_threshold]

    # (4) outliers removing from contour
    outliers = find_outliers(np.abs(np.diff(f0_tonality)), n_quintiles=20)
    f0_tonality_outliers = f0_tonality[2:][outliers]

    x2 = x1[2:][outliers]
    S2 = S1[:, 2:][:, outliers]

    # take only central part as borders are noisy
    t = ignore_border
    k = len(f0_tonality_outliers)
    start = int(t * k)
    stop = int((1 - t) * k)
    mean_f0 = int(np.mean(f0_tonality_outliers[start:stop]))
    max_f0 = int(np.max(f0_tonality_outliers[start:stop]))
    min_f0 = int(np.min(f0_tonality_outliers[start:stop]))

    # (4) contour smoothing by spline interpolation
    tck = sp.interpolate.splrep(
        x2, f0_tonality_outliers / FMAX, k=3, s=smoothness)
    f0_smooth = FMAX * sp.interpolate.splev(x, tck, der=0)

    # (5) compute mean normalize 2nd derivative from normalized contou
    ddy = sp.interpolate.splev(x, tck, der=2)
    # take only central part as borders are noisy
    t = ignore_border
    m = len(ddy)
    no_border_ddy = ddy[int(t * m):int((1 - t) * m)]
    mnddy = np.mean(np.abs(no_border_ddy))

    rms = librosa.feature.rms(S=S2, frame_length=168, hop_length=42).flatten()
    mean_rms = np.mean(rms)

    if plot:
        plot_f0(
            spectrogram, x, f0, x1, f0_tonality,
            x2, f0_tonality_outliers,
            f0_smooth, mnddy, title, image_path
        )

    return [
        min_f0,
        max_f0,
        mean_f0,
        mnddy,
        mean_rms
    ]


def plot_f0(
    spectrogram, x, f0, x1, f0_tonality, x2, f0_tonality_outliers,
    f0_smooth, mnddy, title='', image_path=None):

    # (1) plot STFT
    fig, ax = plt.subplots(figsize=(12, 6))
    img = librosa.display.specshow(
        librosa.amplitude_to_db(spectrogram),
        ax=ax,
        **STFTPLOTPARAMS)
    ax.set(title=title)

    # (2) compute contour = max f0 over STFT

    ax.plot(x, f0, color='r', marker='^', linewidth=0, label='max')
    ax.legend(loc='upper right')

    # (3) tonality thresholding of contour

    ax.plot(
        x1, f0_tonality,
        color='g', marker='.', linewidth=0, label='f0_tonality')
    ax.legend(loc='upper right')

    # (4) outliers removing from contour
    # (4) contour smoothing by spline interpolation

    # (5) compute mean normalize 2nd derivative from normalized contour

    ax.plot(
        x2, f0_tonality_outliers,
        color='b', marker='x', linewidth=0, label='f0_tonality_outliers')
    ax.legend(loc='upper right')

    ax.plot(
        x, f0_smooth,
        color='m', marker='x', linewidth=0, label='f0_smooth')
    ax.legend(loc='upper right')

    plt.title(title)
    plt.tight_layout()

    fig.colorbar(img, ax=ax, format="%+2.f dB")

    if image_path:
        fig.savefig(image_path)
