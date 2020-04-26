# add packkage to ŷthon paths
export PYTHONPATH=$PYTHONPATH:$(pwd)/python

# get ffmpeg and ffprobe binaries paths
export FFMPEG_BINARY=$(which ffmpeg)
export FFMPEG_BINARY=$(which ffprobe)

# set data env variables
export AUDIOVOCANA_BASE_PATH="/path/to/base/folder"
export AUDIOVOCANA_XLSX_FOLDER="/path/to/excel/files/folder"
export AUDIOVOCANA_AUDIO_FOLDER="/path/to/audio/files/folder"
