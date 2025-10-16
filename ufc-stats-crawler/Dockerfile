FROM python:3.10-slim
WORKDIR /app

# Fix old dateparser+regex crash
RUN pip install --no-cache-dir "regex==2021.11.10"

# Crawler deps from the repo
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# API deps
COPY api_app/requirements.txt /app/api_requirements.txt
RUN pip install --no-cache-dir -r /app/api_requirements.txt

# Project files (spiders, settings, web UI, etc.)
COPY . /app
