import modal
import os

# Константи
DATA_ROOT = dataai-toolkit
TOOLKIT_DIR = os.path.join(DATA_ROOT, ai-toolkit)
OUTPUTS_DIR = os.path.join(DATA_ROOT, outputs)
DATASETS_DIR = os.path.join(DATA_ROOT, datasets)

# Створення образу
toolkit_image = (
    modal.Image.debian_slim(python_version=3.11)
    .apt_install(git, wget, curl)
    .run_commands(
        git clone httpsgithub.comostrisai-toolkit rootai-toolkit,
        cd rootai-toolkit && git submodule update --init --recursive
    )
    .pip_install(
        torch==2.5.1,
        torchvision,
        huggingface_hub,
        peft,
        diffusers,
        transformers,
        accelerate,
        safetensors,
        omegaconf,
        gradio==4.44.0,
        Pillow,
        requests
    )
    .run_commands(
        cd rootai-toolkit && pip install -e .
    )
)

# Створення Volume для збереження даних
data_volume = modal.Volume.from_name(ai-toolkit-data, create_if_missing=True)

app = modal.App(ai-toolkit-ui)

@app.function(
    gpu=A100,
    timeout=86400,
    volumes={DATA_ROOT data_volume},
    image=toolkit_image,
)
@modal.web_server(8000)
def run_ui()
    import subprocess
    import sys
    
    # Створення необхідних директорій
    os.makedirs(TOOLKIT_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(DATASETS_DIR, exist_ok=True)
    
    # Запуск AI Toolkit UI
    subprocess.Popen([
        sys.executable,
        rootai-toolkitrun_ui.py,
        --host, 0.0.0.0,
        --port, 8000,
        --share
    ])