FROM nvidia/cuda:10.2-cudnn7-runtime-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive

ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME
RUN apt update -y
RUN apt install -y \ 
                zsh\
                curl\
                wget\
                unzip\
                git\
                libsm6\
                libxext6\
                libxrender-dev
RUN apt-get install software-properties-common -y
RUN apt install python3.7   python3.7-venv python3.7-dev -y
RUN curl https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
RUN python3.7 /tmp/get-pip.py

ENV PATH="/home/${USERNAME}/.local/bin:${PATH}"
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV POETRY_VERSION=1.0.5

RUN pip install --upgrade pip setuptools wheel \
    && pip install "poetry==$POETRY_VERSION"

ENV VIRTUAL_ENV=/home/$USERNAME/.virtualenvs/$PROJECT
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /workspace
COPY src ./src
WORKDIR /workspace/src

RUN python3.7 -m venv $VIRTUAL_ENV
RUN python3.7 -m pip install --upgrade pip
RUN poetry  install -vvv


# RUN chown -R $USERNAME:$USERNAME $VIRTUAL_ENV

# the user we're applying this too (otherwise it most likely install for root)
USER $USERNAME
# terminal colors with xterm
ENV TERM xterm
# set the zsh theme
ENV ZSH_THEME agnoster

# run the installation script  
RUN wget https://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh -O - | zsh || true


ENV DEBIAN_FRONTEND=dialog