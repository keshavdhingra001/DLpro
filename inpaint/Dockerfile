FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
        wget libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /

# Install CPU-only PyTorch first (much smaller than CUDA version)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining requirements
RUN pip install --no-cache-dir -r /requirements.txt

ENV MODEL_URL https://www.dropbox.com/s/gb0yamqa6cnoomy/model_no_sigmoid_lr_all_50.pth?dl=1
RUN wget -O model.state_dict "$MODEL_URL" || echo "Model download failed, will run without model"

COPY . /inpaint
RUN mkdir -p /inpaint/app/static/models/ && \
    (mv model.state_dict /inpaint/app/static/models/ 2>/dev/null || true)

EXPOSE 5000

WORKDIR /inpaint/app
CMD [ \
    "gunicorn", "app:setup_app('static/models/model.state_dict')", \
    "--bind", "0.0.0.0:5000", \
    "--config", "gunicorn_config.py", \
    "--workers", "4" \
]
