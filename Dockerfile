# Use an official PyTorch image with CUDA support
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TZ=Etc/UTC

# Install system dependencies
# RUN python3 -m venv /opt/venv
# ENV PATH="/opt/venv/bin:$PATH"
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    build-essential cmake git wget unzip curl \
    libopencv-dev ffmpeg libgl1 libglib2.0-0 \
    python3 python3-pip python3-dev python-is-python3 \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links to ensure pip command works
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Verify Python and pip installation
RUN python --version && pip --version

# Previous parts of your Dockerfile remain unchanged

# Install Python dependencies (using the --break-system-packages flag)
# Install PyTorch packages with CUDA support
RUN pip install --break-system-packages \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install other dependencies
RUN pip install --break-system-packages \
    opencv-contrib-python-headless numpy matplotlib tqdm \
    pyyaml natsort yt-dlp av

# Try to install less common packages separately
RUN pip install --break-system-packages pyrealsense2 || echo "Failed to install pyrealsense2, continuing anyway"
RUN pip install --break-system-packages lietorch || echo "Failed to install lietorch, continuing anyway"

# Rest of your Dockerfile continues here

# Clone the Multicam-MASt3R-SLAM repository
WORKDIR /workspace
COPY . /workspace

# Initialize git submodules
RUN git submodule update --init --recursive

# Install additional Python dependencies from the project
RUN pip install --break-system-packages -e .

# Expose a port for visualization (if needed)
EXPOSE 8080

# Set the default command
CMD ["python3", "main.py", "--config", "config/base.yaml"]
