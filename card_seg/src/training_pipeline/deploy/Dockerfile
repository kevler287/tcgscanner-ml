FROM ultralytics/ultralytics:latest

WORKDIR /app

# Install Python dependencies
COPY card_seg/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --break-system-packages

# Copy source code
COPY card_seg/training_pipeline/ ./card_seg/training_pipeline/
COPY card_seg/config.py ./card_seg/config.py
COPY common/ ./common/

CMD ["python", "-m", "card_seg.training_pipeline.training_job"]