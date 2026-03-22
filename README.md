# Wind-Power-Forecast---CEEMDAN

## Overview

Esse projeto é baseado no artigo [1] e tem por objetivo implementar otimizações no framework da previsão de energia eólico combinando:

- CEEMDAN (Complete Ensemble Empirical Mode Decomposition with Adaptive Noise).
- EWT (Empirical Wavelet Transform);
- LSTM (Long Short-Term Memory Networks);
- GRU (Gated Recurrent Unit)
- Transformers
- Hilber-Hung transformer.
- ...

De forma a poder aumentar a acurácia do modelo de previsão em uma série temporal não linear e com fortes tendências estocásticas.

## Estrutura do Projeto

```text
Wind-Power-Forecast---CEEMDAN/
│
├── CEEMDAN-EWT-LSTM/        # Hybrid decomposition + LSTM models
├── dataset/                 # Input time-series data
│
├── experiments_france.ipynb # Main experimental notebook
├── experiments_france_1_1.py# Script version of experiments
│
├── VMD.py                   # Variational Mode Decomposition
├── hilbert.py               # Hilbert transform functions
├── myfunctions_france_felipe.py # Auxiliary utilities
│
├── requirements.txt
├── Dockerfile
└── README.md
```

## Instalação

Clone o repositório:

```bash
$ git clone https://github.com/felipemunarol/Wind-Power-Forecast---CEEMDAN.git
$ cd Wind-Power-Forecast---CEEMDAN
```

## Docker

Build:

```bash
$ DOCKER_BUILDKIT=1 docker build --output type=local,dest=./build .
```

Run:

```bash
$ docker run --rm solid2
```

No comando run já é executado o script principal.

## References

[1]  Karijadi, Irene & Chou, Shuo-Yan & Dewabharata, Anindhita, 2023. "Wind power forecasting based on hybrid CEEMDAN-EWT deep learning method," Renewable Energy, Elsevier, vol. 218(C).

## Autor

Felipe Munaro Lima
PhD Researcher – Wind Power Forecasting
Orientador: Natanael Moura Junior
Brazil - RJ
