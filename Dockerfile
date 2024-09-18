FROM python:3.11

LABEL authors="andrew"

WORKDIR /app

COPY . .

RUN apt-get update && \
    apt-get install -y curl build-essential libssl-dev pkg-config

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

ENV PATH="/root/.cargo/bin:${PATH}"

RUN pip install maturin

RUN pip install --no-cache-dir -r requirements.txt

RUN maturin develop

ENV PYTHONPATH="${PYTHONPATH}:/app"

EXPOSE 80

CMD ["python", "main.py"]


