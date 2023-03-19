# -----------------------------------------------------------------------------
#  Stage 1: install pyenv
# -----------------------------------------------------------------------------
FROM ubuntu:20.04 AS pyenv

ENV DEBIAN_FRONTEND=noninteractive

# install pyenv with pyenv-installer
COPY pyenv_dependencies.txt pyenv_dependencies.txt

ENV PYENV_GIT_TAG=v2.3.14

RUN apt-get update && \
    apt-get install -y $(cat pyenv_dependencies.txt)
RUN curl https://pyenv.run | bash
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PYENV_ROOT /root/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

RUN pyenv install 3.8.5 && \
    pyenv global 3.8.5

# -----------------------------------------------------------------------------
#  Stage 2: install audio tool
# -----------------------------------------------------------------------------
FROM ubuntu:20.04 AS audiotools

# install ffmpeg and ffprobe
RUN apt-get update && apt-get install -y ffmpeg

# -----------------------------------------------------------------------------
#  Stage 3: user setup
# -----------------------------------------------------------------------------
FROM ubuntu:20.04

COPY --from=pyenv /root/.pyenv /root/.pyenv
ENV PYENV_ROOT /root/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
ENV PYTHONIOENCODING utf-8
RUN python -m pip install --upgrade pip

COPY --from=audiotools /usr/bin/ffmpeg /usr/bin/ffmpeg
COPY --from=audiotools /usr/bin/ffprobe /usr/bin/ffprobe

# install git & nano
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git && \
    apt-get install -y nano

WORKDIR /usv

ENV FFPROBE_BINARY /usr/bin/ffprobe
ENV FFMPEG_BINARY /usr/bin/ffmpeg

# clone project repo and install dependencies
ARG token
ENV env_token $token

RUN git clone https://${env_token}@github.com/jimenaRL/ultra-sonic-vocalizations.git
WORKDIR /usv/ultra-sonic-vocalizations
RUN git checkout docker --
RUN pip install -r python/requirements.txt
ENV PYTHONPATH /usv/ultra-sonic-vocalizations/python

WORKDIR /usv
RUN git clone https://${env_token}@github.com/jimenaRL/usv-experiments.git

WORKDIR /usv/usv-experiments
RUN git checkout abril2022
WORKDIR /usv/usv-experiments/setup-complexUSV-20230318



