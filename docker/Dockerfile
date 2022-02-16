# An `ARG` declared before a FROM can't be used after a FROM.
# To use it use an ARG instruction without a value after a FROM.
ARG tensorflow_version

# use python3 + TensorFlow 2.3.0
FROM tensorflow/tensorflow:${tensorflow_version}
LABEL maintainer="Keishi Ishihara <keishi.ishihara@karelics.fi>"

ARG tensorflow_version
RUN echo "TensorFlow==$tensorflow_version"
ARG work_dir

# install apt dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
	git \
	vim \
    nano \
	wget \
    htop \
    less \
    graphviz \
    libgl1-mesa-dev \
    iputils-ping \
    net-tools

# install python dependencies
# requirements.txt will be removed after container is up
RUN mkdir $work_dir
ADD requirements.txt $work_dir
WORKDIR $work_dir
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# dont write .pyc files
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get clean && rm -rf /var/lib/apt/lists/*
