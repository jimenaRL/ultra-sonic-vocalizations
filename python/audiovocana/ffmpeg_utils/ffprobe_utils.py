import subprocess
from audiovocana.conf import FFPROBE_BINARY

import json

possible_entries = {
                        "codec_name",
                        "codec_type",
                        "sample_fmt",
                        "sample_rate",
                        "channels",
                        "channel_layout",
                        "bits_per_sample",
                        "duration_ts",
                        "duration",
                        "bits_per_raw_sample"
                    }


def get_ffprobe_entry(audio_filename, entry, entry_type="stream"):
    """
        Get entry with ffprobe
    """
    if entry not in possible_entries:
        e = f"{entry} is not in the set of possible entries. "
        e += f"Must be in: {possible_entries}."
        raise ValueError(e)
    cmd1 = " ".join([
        FFPROBE_BINARY,
        "-v error",
        "-show_entries",
        f"{entry_type,}={entry}",
        "-of default=noprint_wrappers=1:nokey=1",
        audio_filename])
    status, output = subprocess.getstatusoutput(cmd1)
    if status:
        e = "ffprobe command returns with status {status}. Output: {output}."
        raise OSError(e)
    return output


def get_infos(audio_filename):
    """
        Get info about an audio file with ffprobe
    """
    audio_filename = "'"+audio_filename+"'"
    cmd = f"ffprobe -v quiet -print_format json -show_format -show_streams "
    cmd += f"{audio_filename}"
    status, output = subprocess.getstatusoutput(cmd)
    if status:
        raise OSError(
            f"ffprobe command returns with status {status}. Output: {output}.")
    return json.loads(output)


def get_duration(audio_filename):
    """
        Get duration of audio filename in second
    """
    return float(get_ffprobe_entry(audio_filename, "duration"))


def get_sample_rate(audio_filename):
    """
        Get sample rate of audio filename in Hz
    """
    return float(get_ffprobe_entry(audio_filename, "sample_rate"))


def get_channel_number(audio_filename):
    """
        Get number of channels  of audio filename
    """
    return int(get_ffprobe_entry(audio_filename, "channels"))
