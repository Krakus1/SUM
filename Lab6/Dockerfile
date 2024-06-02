FROM python:3.10

WORKDIR /app

# Instalacja niezbędnych pakietów
RUN apt-get -y update && apt-get install -y \
    python3-dev \
    apt-utils \
    build-essential \
    net-tools \
    && rm -rf /var/lib/apt/lists/*

# Instalacja pakietów Pythona
RUN pip3 install -U setuptools
RUN pip3 install \
    cython==3.0.6 \
    numpy==1.26.0 \
    pandas==2.1.3

# Kopiowanie i instalacja zależności z requirements.txt
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Kopiowanie reszty aplikacji
COPY . .

# Ustawienie portu i wyeksponowanie go
ENV PORT 8000
EXPOSE 8000

# Uruchomienie aplikacji
CMD gunicorn -w 3 -k uvicorn.workers.UvicornWorker app:app --bind 0.0.0.0:$PORT

