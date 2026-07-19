FROM pytorch/pytorch:2.12.1-cuda12.6-cudnn9-runtime

WORKDIR /app

# Install Python dependencies
COPY ed_check/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --break-system-packages

# Pre-download ResNet18 ImageNet weights so they're baked into the image
RUN python -c "from torchvision.models import resnet18, ResNet18_Weights; resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)"

# Copy source code
COPY ed_check/src/training_pipeline/ ./ed_check/src/training_pipeline/
COPY ed_check/src/config.py ./ed_check/src/config.py
COPY common/ ./common/

CMD ["python", "-m", "ed_check.src.training_pipeline.deploy.training_job"]