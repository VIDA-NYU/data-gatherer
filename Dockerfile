# syntax=docker/dockerfile:1
FROM --platform=linux/amd64 nvidia/cuda:12.6.3-runtime-ubuntu22.04

WORKDIR /app
ENV PYTHONPATH=/app

# Install Python 3.11, Firefox ESR, uv, and system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends wget curl gnupg ca-certificates \
        python3.11 python3.11-dev \
    && install -d -m 0755 /etc/apt/keyrings \
    && curl -fsSL https://packages.mozilla.org/apt/repo-signing-key.gpg \
        -o /etc/apt/keyrings/packages.mozilla.org.asc \
    && echo "deb [signed-by=/etc/apt/keyrings/packages.mozilla.org.asc] https://packages.mozilla.org/apt mozilla main" \
        > /etc/apt/sources.list.d/mozilla.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends firefox \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.local/bin:$PATH"

# Download and install geckodriver
RUN wget -O /tmp/geckodriver.tar.gz "https://github.com/mozilla/geckodriver/releases/download/v0.34.0/geckodriver-v0.34.0-linux64.tar.gz" \
    && tar -xzf /tmp/geckodriver.tar.gz -C /usr/local/bin \
    && chmod +x /usr/local/bin/geckodriver \
    && rm /tmp/geckodriver.tar.gz

# Install CUDA-enabled PyTorch before other requirements so it isn't overridden
RUN uv pip install --system --no-cache torch --index-url https://download.pytorch.org/whl/cu121

# Copy only requirements first for better caching
COPY requirements.txt .

RUN uv pip install --system --no-cache -r requirements.txt

# Copy the rest of the app
COPY . .

RUN uv pip install --system --no-cache .

EXPOSE 8501

CMD ["streamlit", "run", "/app/ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
