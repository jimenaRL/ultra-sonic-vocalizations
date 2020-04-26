from audiovocana.conf import FFMPEG_BINARY
import subprocess
import re
import os
import numpy as np


def format_time(n_seconds, format_="%d:%02d:%09.6f"):
    m, s = divmod(n_seconds, 60)
    h, m = divmod(m, 60)
    formatted_tima = format_ % (h, m, s)
    return formatted_tima


def extract_channels_and_samplerate(line):

    # Find samplerate
    match = re.search(r'(\d+) hz', line)

    if match:
        samplerate = int(match.group(1))
    else:
        samplerate = 0

    # Channel count.
    match = re.search(r'hz, ([^,]+),', line)
    if match:
        mode = match.group(1)
        if mode == 'stereo':
            n_channels = 2
        else:
            match = re.match(r'(\d+) ', mode)
            if match:
                n_channels = int(match.group(1))
            else:
                n_channels = 1
    else:
        n_channels = 0

    return n_channels, samplerate


def load_audio_file(
    audio_filename,
    start_second=None,
    duration_second=None,
    dtype=np.float,
    block_size=4096*16
):
    """
        Load audio as a numpy array using ffmpeg.
    """

    if not os.path.exists(audio_filename):
        raise ValueError(f"{audio_filename}: no such file.")

    command_args = [FFMPEG_BINARY]
    if start_second:
        start_str = format_time(start_second)
        command_args.extend(['-ss', start_str])

    if duration_second:
        duration_str = format_time(duration_second)
        command_args.extend(['-t', duration_str])

    command_args.extend(['-i', audio_filename, '-f', 's16le', '-'])
    proc = subprocess.Popen(
        command_args,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )

    data_list = []
    while True:
        data = proc.stdout.read(block_size)
        data_list.append(data)
        if not data:
            # Stream closed (EOF).
            break

    data_str = b"".join(data_list)

    while True:
        line = proc.stderr.readline()
        if not line:
            # EOF and data not found.
            raise OSError("stream info not found")

        # In Python 3, result of reading from stderr is bytes.
        if isinstance(line, bytes):
            line = line.decode('utf8', 'ignore')

        line = line.strip().lower()

        if 'no such file' in line:
            raise IOError('file not found')
        elif 'invalid data found' in line:
            raise UnsupportedError()
        elif 'audio:' in line:
            n_channels, samplerate = extract_channels_and_samplerate(line)
            break

    return (
        buf_to_float(data_str, dtype=dtype).reshape(-1, n_channels),
        samplerate
    )


def buf_to_float(x, n_bytes=2, dtype=np.float):
    """Convert an integer buffer to floating point values.
    This is primarily useful when loading integer-valued wav data
    into numpy arrays.

    See Also
    --------
    buf_to_float

    Parameters
    ----------
    x : np.ndarray [dtype=int]
        The integer-valued data buffer

    n_bytes : int [1, 2, 4]
        The number of bytes per sample in `x`

    dtype : numeric type
        The target output type (default: 32-bit float)

    Returns
    -------
    x_float : np.ndarray [dtype=float]
        The input data buffer cast to floating point
    """

    # Invert the scale of the data
    scale = 1./float(1 << ((8 * n_bytes) - 1))

    # Construct the format string
    fmt = '<i{:d}'.format(n_bytes)

    # Rescale and format the data buffer
    return scale * np.frombuffer(x, fmt).astype(dtype)
