FROM python:3.7-slim-buster
WORKDIR /app
COPY . /app
RUN sed -i 's/deb.debian.org/archive.debian.org/g' /etc/apt/sources.list \
    && sed -i '/security.debian.org/d' /etc/apt/sources.list \
    && apt-get update -y

RUN pip install -r requirements.txt
CMD ["python3", "app.py"]