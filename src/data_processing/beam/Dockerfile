# Example Dockerfile for a Python-based Beam application
FROM python:3.8-slim
WORKDIR /code

RUN apt-get update && apt-get install -y \
    gcc \
    libc6-dev

COPY ./requirements.txt /code/requirements.txt

COPY ./resources /code/resources

RUN pip3 install --upgrade pip
RUN pip3 install -r /code/requirements.txt

COPY ./app /code/app

WORKDIR /code/app

#CMD ["python", "app/rightmove_processing.py"]

# Start a shell to keep the container running
CMD ["tail", "-f", "/dev/null"]