import modal
import os

# Константи
DATA_ROOT = "/data/ai-toolkit"
TOOLKIT_DIR = os.path.join(DATA_ROOT, "ai-toolkit")
OUTPUTS_DIR = os.path.join(DATA_ROOT, "outputs")
DATASETS_DIR = os.path.join(DATA_ROOT, "datasets")

# Створення образу
toolkit_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget", "curl", "libgl1", "libglib2.0-0")
    .run_commands(
        "git clone https://github.com/ostris/ai-toolkit /root/ai-toolkit",
        "cd /root/ai-toolkit && git submodule update --init --recursive",
    )
    .pip_install(
        "torch==2.5.1",
        "torchvision",
        "huggingface_hub",
        "peft",
        "diffusers",
        "transformers",
        "accelerate",
        "safetensors",
        "omegaconf",
        "gradio==4.44.0",
        "Pillow",
        "requests",
        "wandb",
        "tensorboard",
    )
    .run_commands(
        "cd /root/ai-toolkit && pip install -r requirements.txt || true",
        "cd /root/ai-toolkit && pip install -e .",
    )
)

# Створення Volume для збереження даних
data_volume = modal.Volume.from_name("ai-toolkit-data", create_if_missing=True)

app = modal.App("ai-toolkit-ui")

@app.function(
    gpu="A100",
    timeout=86400,
    volumes={DATA_ROOT: data_volume},
    image=toolkit_image,
    allow_concurrent_inputs=10,
)
@modal.web_server(8000, startup_timeout=300)
def run_ui():
    import subprocess
    import sys
    import time

    # Створення необхідних директорій
    os.makedirs(TOOLKIT_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(DATASETS_DIR, exist_ok=True)

    # Копіюємо ai-toolkit в data volume для збереження
    if not os.path.exists(os.path.join(TOOLKIT_DIR, "run_ui.py")):
        subprocess.run(
            ["cp", "-r", "/root/ai-toolkit/*", TOOLKIT_DIR],
            shell=True,
            check=False
        )

    # Запуск AI Toolkit UI
    cmd = [
        sys.executable,
        "/root/ai-toolkit/run_ui.py",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--output-dir", OUTPUTS_DIR,
        "--config-dir", TOOLKIT_DIR,
    ]

    subprocess.Popen(cmd)

    # Чекаємо поки сервер запуститься
    time.sleep(10)
