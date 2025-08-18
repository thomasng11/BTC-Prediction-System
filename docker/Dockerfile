# Use PyTorch CUDA base image with Python 3.10
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# Install system dependencies for SQLite, Chrome, and ChromeDriver
RUN apt-get update && apt-get install -y \
    sqlite3 \
    libsqlite3-dev \
    wget \
    gnupg \
    unzip \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Install Google Chrome
RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - \
    && echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update \
    && apt-get install -y google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*

# Install ChromeDriver
RUN CHROMEDRIVER_VERSION=$(wget -qO- https://googlechromelabs.github.io/chrome-for-testing/LATEST_RELEASE_STABLE) \
    && wget -q https://storage.googleapis.com/chrome-for-testing-public/$CHROMEDRIVER_VERSION/linux64/chromedriver-linux64.zip \
    && unzip chromedriver-linux64.zip \
    && mv chromedriver-linux64/chromedriver /usr/local/bin/ \
    && chmod +x /usr/local/bin/chromedriver \
    && rm chromedriver-linux64.zip

# Set working directory inside the container
WORKDIR /app

# Copy requirements.txt from docker/ and install dependencies
COPY docker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/

# Expose port for Streamlit
EXPOSE 8501

# Default command (overridden in docker-compose.yml)
CMD ["python", "src/pipeline.py"]