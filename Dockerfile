FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required for OpenCV and other packages
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the models in CPU mode
RUN python -c 'from easyocr import Reader; Reader(["de", "fr", "it"], gpu=False)'

# Copy the application code
COPY . .

# Create directories if they don't exist
RUN mkdir -p detected_images  detected_results ids card_data

# Expose the port the app runs on
EXPOSE 8000

# Command to run the API
CMD ["python", "api.py"]