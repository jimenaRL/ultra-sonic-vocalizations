# add packkage to ŷthon paths
export PYTHONPATH=$PYTHONPATH:$(pwd)/python

# get ffmpeg and ffprobe binaries paths
export FFMPEG_BINARY=$(which ffmpeg)
export FFPROBE_BINARY=$(which ffprobe)