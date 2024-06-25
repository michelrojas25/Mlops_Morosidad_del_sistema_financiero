# Base Image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements.txt
COPY requerimientos.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requerimientos.txt

# Copy the rest of the code
COPY . .

# Define entrypoint
ENTRYPOINT ["python", "train_model.py"]

