FROM ultralytics/ultralytics:latest

WORKDIR /app

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e . --break-system-packages

# Copy source code
COPY card_seg/training_pipeline/ ./card_seg/training_pipeline/
COPY card_seg/__init__.py ./card_seg/__init__.py
COPY card_seg/config.py ./card_seg/config.py
COPY config/ ./config/

CMD ["python", "-m", "card_seg.training_pipeline.training_job"]