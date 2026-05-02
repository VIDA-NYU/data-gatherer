# syntax=docker/dockerfile:1
FROM --platform=linux/amd64 nvidia/cuda:12.6.3-runtime-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      wget xz-utils ca-certificates \
      python3.11 python3.11-dev python3-pip \
      libgtk-3-0 libdbus-glib-1-2 libasound2 libx11-xcb1 libxcomposite1 \
      libxrender1 libxrandr2 libxtst6 libglib2.0-0 libgconf-2-4 libnss3 \
      libgdk-pixbuf2.0-0 libxss1 libpangocairo-1.0-0 fonts-liberation && \
    rm -rf /var/lib/apt/lists/* && \
    wget -O /tmp/firefox.tar.xz "https://download.mozilla.org/?product=firefox-latest&os=linux64&lang=en-US" && \
    tar -xJf /tmp/firefox.tar.xz -C /opt && \
    ln -sf /opt/firefox/firefox /usr/bin/firefox && \
    rm /tmp/firefox.tar.xz

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