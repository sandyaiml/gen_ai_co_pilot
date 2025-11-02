# Use lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy dependency file first
COPY requirements.txt .

# Install system dependencies (required for OpenCV, Plotly, Matplotlib rendering)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        fonts-dejavu-core \
        && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app
COPY . .

# Make the startup script executable
RUN chmod +x /app/start.sh

# Streamlit environment settings
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8080
ENV PORT=8080

# Expose port
EXPOSE 8080

# Start app
CMD ["/app/start.sh"]
