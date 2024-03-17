FROM python:3.11

LABEL authors="andrew"

# Set the working directory in the container
WORKDIR /app

# Copy everything from the context into the /app directory
COPY . .

RUN python3 -m venv /opt/venv

RUN apt-get update
RUN apt-get install -y python3
RUN apt-get install -y python3-pip
RUN /opt/venv/bin/python -m pip install --no-cache-dir -r requirements.txt


ENV PYTHONPATH "${PYTHONPATH}:/app"

EXPOSE 80




CMD ["/opt/venv/bin/python", "main.py"]


