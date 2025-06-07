FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt file
COPY requirements.txt /tmp/

# Install PyTorch CPU-only version
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install dependencies from requirements.txt
RUN pip install -r /tmp/requirements.txt

RUN pip install opencv-python-headless

RUN pip install flask-cors

ENV APP_ROOT=/opt/sam2
ENV PYTHONUNBUFFERED=1
ENV SAM2_BUILD_CUDA=0

# Create directory for checkpoints
RUN mkdir -p ${APP_ROOT}/checkpoints

ADD https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt ${APP_ROOT}/checkpoints/sam2.1_hiera_small.pt

CMD ["python3"]