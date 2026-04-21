# Commands: 
#   - docker build -t ceemdan-model:v1 .
#   - executa em cpu: docker run -it -v ${PWD}:/app ceemdan-model:v1
#   - executa em gpu: docker run -it --gpus all -v ${PWD}:/app ceemdan-model:v1

FROM tensorflow/tensorflow:2.15.0-gpu

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

# NÃO incluir tensorflow no requirements
RUN pip install -r requirements.txt

COPY . .

CMD ["python3", "run_france.py"]