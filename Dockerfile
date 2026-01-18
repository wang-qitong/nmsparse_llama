FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    bzip2 \
    build-essential \
    ninja-build \
    cmake \
    pkg-config \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

ENV CONDA_DIR=/opt/conda
RUN curl -fsSL -o /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py312_24.7.1-0-Linux-x86_64.sh \
    && bash /tmp/miniconda.sh -b -p ${CONDA_DIR} \
    && rm -f /tmp/miniconda.sh

ENV PATH=${CONDA_DIR}/bin:${PATH}
SHELL ["/bin/bash", "-lc"]

WORKDIR /wangqitong

COPY myenv_environment.yml /tmp/myenv_environment.yml
RUN conda env create -f /tmp/myenv_environment.yml \
    && conda clean -a -y \
    && rm -f /tmp/myenv_environment.yml

ENV CONDA_DEFAULT_ENV=myenv
ENV PATH=${CONDA_DIR}/envs/myenv/bin:${PATH}

ENV CUDA_HOME=/usr/local/cuda-12.4
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

ENV TORCH_EXTENSIONS_DIR=/wangqitong/.torch_extensions
ENV TMPDIR=/wangqitong/.tmp
RUN mkdir -p /wangqitong/.torch_extensions /wangqitong/.tmp

COPY . /wangqitong

CMD ["bash"]
