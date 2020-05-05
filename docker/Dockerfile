FROM ubuntu:18.04

# install miniconda
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# update conda
RUN conda update -n base -c defaults conda

# install jupyter notebook
RUN conda install -y jupyter

# install git & nano
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git && \
    apt-get install -y nano

# install pip
RUN conda install -y -c anaconda pip && conda update pip

# clone project repo and install dependencies
RUN git clone https://github.com/jimenaRL/ultra-sonic-vocalizations.git

RUN pip install -r ultra-sonic-vocalizations/python/requirements.txt



