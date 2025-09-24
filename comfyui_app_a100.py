import os
import shutil
import subprocess
from typing import Optional
from huggingface_hub import hf_hub_download

# Paths
DATA_ROOT = "/data/comfy"
DATA_BASE = os.path.join(DATA_ROOT, "ComfyUI")
CUSTOM_NODES_DIR = os.path.join(DATA_BASE, "custom_nodes")
MODELS_DIR = os.path.join(DATA_BASE, "models")
TMP_DL = "/tmp/download"

# ComfyUI default install location
DEFAULT_COMFY_DIR = "/root/comfy/ComfyUI"

def git_clone_cmd(node_repo: str, recursive: bool = False, install_reqs: bool = False) -> str:
    name = node_repo.split("/")[-1]
    dest = os.path.join(DEFAULT_COMFY_DIR, "custom_nodes", name)
    cmd = f"git clone https://github.com/{node_repo} {dest}"
    if recursive:
        cmd += " --recursive"
    if install_reqs:
        cmd += f" && pip install -r {dest}/requirements.txt"
    return cmd

def hf_download(subdir: str, filename: str, repo_id: str, subfolder: Optional[str] = None):
    out = hf_hub_download(repo_id=repo_id, filename=filename, subfolder=subfolder, local_dir=TMP_DL)
    target = os.path.join(MODELS_DIR, subdir)
    os.makedirs(target, exist_ok=True)
    shutil.move(out, os.path.join(target, filename))

import modal

# Build image with ComfyUI installed to default location /root/comfy/ComfyUI
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "wget", "libgl1-mesa-glx", "libglib2.0-0", "ffmpeg")
    .run_commands([
        "pip install --upgrade pip",
        "pip install --no-cache-dir comfy-cli uv",
        "uv pip install --system --compile-bytecode huggingface_hub[hf_transfer]==0.28.1",
        # Install ComfyUI to default location
        "comfy --skip-prompt install --nvidia"
    ])
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# Install nodes to default ComfyUI location during build
image = image.run_commands([
    "comfy node install rgthree-comfy comfyui-impact-pack comfyui-impact-subpack ComfyUI-YOLO comfyui-inspire-pack comfyui_ipadapter_plus wlsh_nodes ComfyUI_Comfyroll_CustomNodes comfyui_essentials ComfyUI-GGUF"
])

# Git-based nodes baked into image at default ComfyUI location
for repo, flags in [
    ("ssitu/ComfyUI_UltimateSDUpscale", {'recursive': True}),
    ("welltop-cn/ComfyUI-TeaCache", {'install_reqs': True}),
    ("nkchocoai/ComfyUI-SaveImageWithMetaData", {}),
    ("receyuki/comfyui-prompt-reader-node", {'recursive': True, 'install_reqs': True}),
]:
    image = image.run_commands([git_clone_cmd(repo, **flags)])

# FLUX Model download tasks
flux_model_tasks = [
    ("unet/FLUX", "flux1-dev-Q8_0.gguf", "city96/FLUX.1-dev-gguf", None),
    ("clip/FLUX", "t5-v1_1-xxl-encoder-Q8_0.gguf", "city96/t5-v1_1-xxl-encoder-gguf", None),
    ("clip/FLUX", "clip_l.safetensors", "comfyanonymous/flux_text_encoders", None),
    ("checkpoints", "flux1-dev-fp8-all-in-one.safetensors", "camenduru/FLUX.1-dev", None),
    ("loras", "mjV6.safetensors", "strangerzonehf/Flux-Midjourney-Mix2-LoRA", None),
    ("vae/FLUX", "ae.safetensors", "ffxvs/vae-flux", None),
]

# Qwen-Image-Edit Model download tasks (NEW)
qwen_model_tasks = [
    ("diffusion_models", "qwen_image_edit_fp8_e4m3fn.safetensors", "Comfy-Org/Qwen-Image-Edit_ComfyUI", None),
    ("text_encoders", "qwen_2.5_vl_7b_fp8_scaled.safetensors", "Comfy-Org/Qwen-Image_ComfyUI", None),
    ("vae", "qwen_image_vae.safetensors", "Comfy-Org/Qwen-Image_ComfyUI", None),
    ("loras", "Qwen-Image-Lightning-4steps-V1.0.safetensors", "lightx2v/Qwen-Image-Lightning", None),
    ("loras", "Qwen-Image-Lightning-8steps-V1.0.safetensors", "lightx2v/Qwen-Image-Lightning", None),
    # Optional GGUF version for lower VRAM usage
    ("unet", "Q8_0.gguf", "QuantStack/Qwen-Image-Edit-2509-GGUF", None),
]

# Combine all model tasks
model_tasks = flux_model_tasks + qwen_model_tasks

extra_cmds = [
    f"wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth -P {MODELS_DIR}/upscale_models",
]

# Create volume
vol = modal.Volume.from_name("comfyui-app", create_if_missing=True)

app = modal.App(name="comfyui", image=image)

@app.function(
    max_containers=1,
    scaledown_window=300,
    timeout=1800,
    gpu="A100",
    volumes={DATA_ROOT: vol},
)

@modal.concurrent(max_inputs=10)
@modal.web_server(8000, startup_timeout=300)
def ui():
    # Check if volume is empty (first run)
    if not os.path.exists(os.path.join(DATA_BASE, "main.py")):
        print("First run detected. Copying ComfyUI from default location to volume...")
        # Ensure DATA_ROOT exists
        os.makedirs(DATA_ROOT, exist_ok=True)
        
        # Copy ComfyUI from default location to volume
        if os.path.exists(DEFAULT_COMFY_DIR):
            print(f"Copying {DEFAULT_COMFY_DIR} to {DATA_BASE}")
            subprocess.run(f"cp -r {DEFAULT_COMFY_DIR} {DATA_ROOT}/", shell=True, check=True)
        else:
            print(f"Warning: {DEFAULT_COMFY_DIR} not found, creating empty structure")
            os.makedirs(DATA_BASE, exist_ok=True)

    # [Весь твій існуючий код для update ComfyUI backend, Manager, pip, etc...]
    # ... (код залишається незмінним до секції download models)

    # Ensure all required directories exist (INCLUDING NEW QWEN DIRECTORIES)
    required_dirs = [
        CUSTOM_NODES_DIR, 
        MODELS_DIR,
        os.path.join(MODELS_DIR, "diffusion_models"),  # For Qwen main model
        os.path.join(MODELS_DIR, "text_encoders"),     # For Qwen text encoder
        os.path.join(MODELS_DIR, "unet"),              # For GGUF version
        TMP_DL
    ]
    
    for d in required_dirs:
        os.makedirs(d, exist_ok=True)

    # Download models at runtime (only if missing) - NOW INCLUDES QWEN
    print("Checking and downloading missing FLUX and Qwen-Image-Edit models...")
    for sub, fn, repo, subf in model_tasks:
        target = os.path.join(MODELS_DIR, sub, fn)
        if not os.path.exists(target):
            print(f"Downloading {fn} to {target}...")
            try:
                hf_download(sub, fn, repo, subf)
                print(f"Successfully downloaded {fn}")
            except Exception as e:
                print(f"Error downloading {fn}: {e}")
        else:
            print(f"Model {fn} already exists, skipping download")

    # Run extra download commands
    print("Running additional downloads...")
    for cmd in extra_cmds:
        try:
            print(f"Running: {cmd}")
            result = subprocess.run(cmd, shell=True, check=False, cwd=DATA_BASE, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"Command completed successfully")
            else:
                print(f"Command failed with return code {result.returncode}: {result.stderr}")
        except Exception as e:
            print(f"Error running command {cmd}: {e}")

    # Set COMFY_DIR environment variable to volume location
    os.environ["COMFY_DIR"] = DATA_BASE

    # Launch ComfyUI from volume location
    print(f"Starting ComfyUI from {DATA_BASE} with FLUX and Qwen-Image-Edit support...")
    
    # Start ComfyUI server with correct syntax and latest frontend
    cmd = ["comfy", "launch", "--", "--listen", "0.0.0.0", "--port", "8000", "--front-end-version", "Comfy-Org/ComfyUI_frontend@latest"]
    print(f"Executing: {' '.join(cmd)}")
    
    process = subprocess.Popen(
        cmd,
        cwd=DATA_BASE,
        env=os.environ.copy()
    )
    
    process.wait()
