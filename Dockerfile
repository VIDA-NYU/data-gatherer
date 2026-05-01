# syntax=docker/dockerfile:1
FROM --platform=linux/amd64 nvidia/cuda:12.6.3-runtime-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends wget firefox python3.11 python3.11-dev python3-pip \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Download and install geckodriver
RUN wget -O /tmp/geckodriver.tar.gz "https://github.com/mozilla/geckodriver/releases/download/v0.36.0/geckodriver-v0.36.0-linux64.tar.gz" \
    && tar -xzf /tmp/geckodriver.tar.gz -C /usr/local/bin \
    && chmod +x /usr/local/bin/geckodriver \
    && rm /tmp/geckodriver.tar.gz

# Create writable Firefox profile directory for container compatibility
RUN mkdir -p /tmp/firefox_profile && chmod 777 /tmp/firefox_profile

# Copy requirements and install
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

ENV PYTHONPATH=/app
EXPOSE 8501
CMD ["python3", "-m", "streamlit", "run", "/app/ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]