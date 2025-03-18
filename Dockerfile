# Use the official TensorFlow CPU image (check Docker Hub for latest tags)
FROM tensorflow/tensorflow:2.9.1

# Create a working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt

# Copy your training script into the container
COPY train_cifar10.py /app

# Default command to run the script
CMD ["python", "train_cifar10.py"]
