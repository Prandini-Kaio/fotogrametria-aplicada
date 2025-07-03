FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Instala dependências necessárias
RUN apt update && apt install -y \
    git cmake g++ libboost-all-dev libeigen3-dev \
    libopencv-dev libglew-dev libglfw3-dev libpng-dev libjpeg-dev libtiff-dev \
    libatlas-base-dev libsuitesparse-dev libvtk7-dev libopenexr-dev \
    libprotobuf-dev protobuf-compiler qtbase5-dev libqt5opengl5-dev \
    libceres-dev ninja-build wget unzip \
    && rm -rf /var/lib/apt/lists/*

# Clona o repositório do OpenMVS
WORKDIR /opt
RUN git clone https://github.com/cdcseacave/openMVS.git

WORKDIR /opt/openMVS

# Checkout para versão estável (opcional)
RUN git checkout v2.0

# Inicializa submódulos corretamente
RUN git submodule update --init --recursive

WORKDIR /opt/openMVS/build
RUN cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/opt/openmvs_bin \
    -DOpenCV_DIR=/usr/lib/x86_64-linux-gnu/cmake/opencv4 && \
    cmake --build . --target install -- -j$(nproc)

# Adiciona OpenMVS ao PATH
ENV PATH="/opt/openmvs_bin/bin:$PATH"

WORKDIR /data
