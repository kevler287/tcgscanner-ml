FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e . --break-system-packages

# Pre-download ResNet18 ImageNet weights so they're baked into the image
RUN python -c "from torchvision.models import resnet18, ResNet18_Weights; resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)"

# Copy source code
COPY ed_check/training_pipeline/ ./ed_check/training_pipeline/
COPY ed_check/__init__.py ./ed_check/__init__.py
COPY ed_check/config.py ./ed_check/config.py
COPY common/ ./common/

CMD ["python", "-m", "ed_check.training_pipeline.deploy.training_job"]