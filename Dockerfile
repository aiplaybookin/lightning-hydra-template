FROM python:3.9.13-slim

# Basic setup
RUN apt update
RUN apt install -y \
    build-essential \
    git \
    curl \
    ca-certificates \
    wget \
    && rm -rf /var/lib/apt/lists

# Set working directory
WORKDIR /workspace/project

# Install requirements
COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt \
    && rm requirements.txt
