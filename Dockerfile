# Commands: 
#   - docker build -t ceemdan-model:v1 .
#   - executa em cpu: docker run -it -v "${PWD}:/app" ceemdan-model:v1
#   - executa em gpu: docker run -it --gpus all -v "${PWD}:/app" ceemdan-model:v1

FROM tensorflow/tensorflow:2.3.0-gpu

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements_docker.txt .

# NÃO inclui o tensorflow no requirements
RUN pip install -r requirements_docker.txt

COPY . .

CMD ["python3", "run_france.py"]