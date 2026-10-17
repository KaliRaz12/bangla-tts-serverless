FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app
RUN pip install --no-cache-dir runpod soundfile transformers==4.46.1 \
    git+https://github.com/huggingface/parler-tts.git

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
