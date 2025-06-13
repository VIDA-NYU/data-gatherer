# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . /app

# Install the package in editable mode
RUN pip install --no-cache-dir -e .

# Install wget to download geckodriver
RUN apt-get update && apt-get install -y wget firefox-esr

# Download Linux geckodriver during build
RUN wget -O /tmp/geckodriver.tar.gz "https://github.com/mozilla/geckodriver/releases/download/v0.34.0/geckodriver-v0.34.0-linux64.tar.gz" \
    && tar -xzf /tmp/geckodriver.tar.gz -C /usr/local/bin \
    && chmod +x /usr/local/bin/geckodriver \
    && rm /tmp/geckodriver.tar.gz

# Then use /usr/local/bin/geckodriver as your driver_path

# Expose Streamlit default port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "/app/ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]