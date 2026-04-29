# syntax=docker/dockerfile:1
FROM --platform=linux/amd64 nvidia/cuda:12.1.0-runtime-ubuntu22.04

WORKDIR /app

# Install Python 3.11, Firefox ESR (via Mozilla PPA), and system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends software-properties-common wget \
    && add-apt-repository ppa:mozillateam/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3-pip \
        firefox-esr \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Download and install geckodriver
RUN wget -O /tmp/geckodriver.tar.gz "https://github.com/mozilla/geckodriver/releases/download/v0.34.0/geckodriver-v0.34.0-linux64.tar.gz" \
    && tar -xzf /tmp/geckodriver.tar.gz -C /usr/local/bin \
    && chmod +x /usr/local/bin/geckodriver \
    && rm /tmp/geckodriver.tar.gz

# Install CUDA-enabled PyTorch before other requirements so it isn't overridden
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121

# Copy only requirements first for better caching
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

RUN pip install --no-cache-dir .

EXPOSE 8501

CMD ["streamlit", "run", "/app/ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]