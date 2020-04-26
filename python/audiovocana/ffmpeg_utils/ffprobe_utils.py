import subprocess
from dzr_utils.conf import FFPROBE_BINARY

import json

possible_entries =  {
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
        raise ValueError("%s is not in the set of possible entries. Must be in: %s"%(entry, str(possible_entries)))
    status,output = subprocess.getstatusoutput(" ".join([FFPROBE_BINARY, "-v error", "-show_entries","%s=%s"%(entry_type,entry),"-of default=noprint_wrappers=1:nokey=1",audio_filename]))
    if status:
        raise OSError("ffprobe command returns with status %d. Output: %s"%(status, output))
    return output

def get_infos(audio_filename):
    """
        Get info about an audio file with ffprobe
    """
    audio_filename = "'"+audio_filename+"'"
    status,output = subprocess.getstatusoutput("ffprobe -v quiet -print_format json -show_format -show_streams %s"%audio_filename)
    if status:
        raise OSError("ffprobe command returns with status %d. Output: %s"%(status, output))
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

