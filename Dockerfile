FROM python:3.10-slim-bullseye

# variáveis de ambiente necessárias
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CUDA_PATH=/usr/local/cuda
ENV QT_QPA_PLATFORM=offscreen

# Instala libs do sistema necessárias (imagem slim)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    ca-certificates \
    libgl1-mesa-glx \
    libgtk2.0-dev \
    libboost-all-dev \
    && rm -rf /var/lib/apt/lists/*

# (Opcional) Instruções para instalar COLMAP/OpenMVG/OpenMVS
# Recomendo preparar uma imagem base separada que já contenha os binários.
# Exemplo (comentado): RUN apt-get install -y colmap

WORKDIR /app

# Copia requirements e instala dependências Python
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# Copia código
COPY . /app

# Cria diretórios padrão
RUN mkdir -p /app/resources/input /app/resources/output /app/notebooks

# Usuário não-root opcional (preserva permissões)
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -g ${GROUP_ID} appgroup || true && \
    useradd -m -u ${USER_ID} -g ${GROUP_ID} appuser || true
USER appuser

# Porta para Jupyter (se necessário)
EXPOSE 8888

# Entrypoint por padrão (pode ser sobrescrito no compose)
CMD ["python3", "run_pipeline.py"]
