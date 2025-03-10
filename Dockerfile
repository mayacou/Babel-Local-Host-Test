# Use a base image with Python and necessary tools
FROM python:3.9-slim

# Set environment variables for non-interactive installations
ENV DEBIAN_FRONTEND=noninteractive

# Install git, curl, and other dependencies
RUN apt-get update && \
    apt-get install -y git curl && \
    pip install --upgrade pip && \
    apt-get clean

# Set up working directory inside the container
WORKDIR /app

# Copy your code into the container
COPY . /app

# Install HuggingFace CLI (to enable HuggingFace login)
RUN pip install --no-cache-dir huggingface_hub

# Set up virtual environment (optional, you can install globally if preferred)
RUN python3 -m venv venv

# Set environment variables to use the virtual environment
ENV PATH="/app/venv/bin:$PATH"

# Install project dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run the script (replace with your specific command)
CMD ["python3", "test_all_models.py"]
