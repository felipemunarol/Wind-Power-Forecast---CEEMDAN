# Commands: 
#   - docker build -t ceemdan-model:v1 .
#   - docker run -it -v ${PWD}:/app wind-model

FROM python:3.7.16-slim-bullseye

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Dependências de sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copia requirements primeiro (cache eficiente)
COPY requirements.txt .

# Atualiza pip
RUN pip install --upgrade pip

# PyTorch CPU-only (COMANDO OFICIAL)
RUN pip3 install torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Demais dependências Python
RUN pip install -r requirements.txt

# Código do projeto
COPY . .

# Comando
# Comando padrão
# CMD ["python3"]
CMD ["python3", "experiments_france_1_1.py"]
