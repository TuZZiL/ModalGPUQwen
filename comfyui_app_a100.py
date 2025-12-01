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
    ("loras", "mjV6.safetensors", "strangerzonehf/Flux-Midjourney-Mix2-LoRA", None),
    ("vae/FLUX", "ae.safetensors", "ffxvs/vae-flux", None),
]

# Qwen-Image-Edit Model download tasks (CORRECTED PATHS)
qwen_model_tasks = [
    # Main Qwen-Image-Edit model - в підпапці split_files/diffusion_models
    ("diffusion_models", "qwen_image_fp8_e4m3fn.safetensors", "Comfy-Org/Qwen-Image_ComfyUI", "split_files/diffusion_models"),
    ("diffusion_models", "qwen_image_edit_2509_fp8_e4m3fn.safetensors", "Comfy-Org/Qwen-Image-Edit_ComfyUI", "split_files/diffusion_models"),
    ("diffusion_models", "z_image_turbo_bf16.safetensors", "Comfy-Org/z_image_turbo", "split_files/diffusion_models"),
    # Text encoder - в головній папці Qwen-Image_ComfyUI
    ("text_encoders", "qwen_2.5_vl_7b_fp8_scaled.safetensors", "Comfy-Org/Qwen-Image_ComfyUI", "split_files/text_encoders"),
    ("text_encoders", "Josiefied-Qwen3-8B-abliterated-v1.Q8_0.gguf", "mradermacher/Josiefied-Qwen3-8B-abliterated-v1-GGUF", ""),
    ("text_encoders", "qwen_3_4b.safetensors", "Comfy-Org/z_image_turbo", "split_files/text_encoders"),
    ("text_encoders", "qwen-4b-zimage-heretic-q8.gguf", "Lockout/qwen3-4b-heretic-zimage", None),

    # VAE model - в головній папці Qwen-Image_ComfyUI  
    ("vae", "qwen_image_vae.safetensors", "Comfy-Org/Qwen-Image_ComfyUI", "split_files/vae"),
    ("vae", "ae.safetensors", "Comfy-Org/z_image_turbo", "split_files/vae"),
    
    # Lightning LoRA models
    ("loras", "Qwen-Image-Lightning-4steps-V1.0.safetensors", "ModelTC/Qwen-Image-Lightning", None),
    ("loras", "Qwen-Image-Lightning-8steps-V1.0.safetensors", "ModelTC/Qwen-Image-Lightning", None),
    ("loras", "Qwen-Image-Lightning-4steps-V2.0.safetensors", "lightx2v/Qwen-Image-Lightning", None),
    ("loras", "Qwen-Image-Lightning-8steps-V2.0.safetensors", "lightx2v/Qwen-Image-Lightning", None),
    ("loras", "Qwen-Image-Edit-2509-Lightning-8steps-V1.0-fp32.safetensors", "andrewwe/qwLoras", None),

    # Додаткові Qwen LoRA моделі з HuggingFace
    ("loras", "qwen-studio-realism.safetensors", "prithivMLmods/Qwen-Image-Studio-Realism", None),
    ("loras", "qwen_image_nsfw.safetensors", "starsfriday/Qwen-Image-NSFW", None),
    ("loras", "2111206_removeclothing_qwen-edit.safetensors", "andrewwe/qwLoras", None),
    ("loras", "bumpynipples1_qwen.safetensors", "andrewwe/qwLoras", None),
    ("loras", "qwen_image_edmannequin-clipper_v1.0.safetensors", "andrewwe/qwLoras", None),
    ("loras", "qwen_image_edit_remove-clothing_v1.0.safetensors", "andrewwe/qwLoras", None),
    ("loras", "2114841_qwen_edit_nsfw.safetensors", "andrewwe/qwLoras", None),
    ("loras", "Facial_Cumshots_For_Qwen_Image_V1.safetensors", "andrewwe/qwLoras", None),
    ("loras", "2066568_qwen_edit_uncenudify_lora_v3.safetensors", "andrewwe/qwLoras", None),
    ("loras", "Putithere_Qwen edit_V2.0.safetensors", "andrewwe/qwLoras", None),
    ("loras", "QWEN_JTitsT2_5.safetensors", "andrewwe/qwLoras", None),
    ("loras", "Qwen-Image-Edit-Lowres-Fix.safetensors", "andrewwe/qwLoras", None),
    ("loras", "pose_transfer_v2_qwen_edit.safetensors", "andrewwe/qwLoras", None),
    ("loras", "p0ssy_lora_v1qwenedit.safetensors", "andrewwe/qwLoras", None),
    ("loras", "consistence_edit_v1.safetensors", "andrewwe/qwLoras", None),
    ("loras", "qwenOUTFITootd_colour-19-3600.safetensors", "andrewwe/qwLoras", None),
    ("loras", "InSubject-0.5.safetensors", "peteromallet/Qwen-Image-Edit-InSubject", None),
    ("loras", "qwen-edit-remover.safetensors", "starsfriday/Qwen-Image-Edit-Remover-General-LoRA", None),
    ("loras", "InStyle-0.5.safetensors", "peteromallet/Qwen-Image-Edit-InStyle", None),
    ("loras", "consistence_edit_v1.safetensors", "andrewwe/qwLoras", None),
    ("loras", "QWEN_ed_removed_my1.safetensors", "andrewwe/qwLoras", None),
    ("loras", "QWEN_ed_removed_my1_000001500.safetensors", "andrewwe/qwLoras", None),
    ("loras", "milk_juggs_QWEN.safetensors", "andrewwe/qwLoras", None),
    ("loras", "muscle_women_QWEN.safetensors", "andrewwe/qwLoras", None),
    ("loras", "qwen-image-edit-2509-inscene-lora.safetensors", "andrewwe/qwLoras", None),
    ("loras", "qwen_image_snapchat.safetensors", "andrewwe/qwLoras", None),
    ("loras", "Lora_Qwen-Real_perfect_sex.safetensors", "andrewwe/qwLoras", None),
    ("loras", "breast_slider_qwen_v1.safetensors", "andrewwe/qwLoras", None),
    ("loras", "hips_size_slider_v1qwen.safetensors", "andrewwe/qwLoras", None),
    ("loras", "QWEN_ed_removed_my2.safetensors", "andrewwe/qwLoras", None),
    ("loras", "QwennBustyLoraMy.safetensors", "andrewwe/qwLoras", None),
    ("loras", "the20cleavage_qwen.safetensors", "andrewwe/qwLoras", None),
    ("loras", "editpicforpartV1-2.0.safetensors", "andrewwe/qwLoras", None),
    ("loras", "big_nipples_QWEN.safetensors", "andrewwe/qwLoras", None),
    ("loras", "Nipple_EnhancerQwe_BreastsLoRA_Epoch60.safetensors", "andrewwe/qwLoras", None),
    ("loras", "QwenSnofs1_1.safetensors", "andrewwe/qwLoras", None),
    ("loras", "RealBreastNipples-QWEN-rbn-GMRqwen.safetensors", "andrewwe/qwLoras", None),
    ("loras", "qwen_edit_2509_ObjectRemovalAlpha.safetensors", "andrewwe/qwLoras", None),
    ("loras", "Accelerator-QwenImage-Lightning-8steps-PAseer.safetensors", "andrewwe/qwLoras", None),
    ("loras", "next-scene_lora_v1-3000qwen.safetensors", "andrewwe/qwLoras", None),
    ("loras", "1GIRL_QWEN_V2.safetensors", "andrewwe/qwLoras", None),
    ("loras", "Human_Focus_Photography_v1.0.safetensors", "andrewwe/qwLoras", None),
    ("loras", "Qwen-MysticXXX-v1.safetensors", "andrewwe/qwLoras", None),
    ("loras", "Qwen-iPhone-V1.1.safetensors", "andrewwe/qwLoras", None),
    ("loras", "Samsung.safetensors", "andrewwe/qwLoras", None),
    ("loras", "Qwen-Image_SmartphonePhotoReality_v4_TRIGGERamateur photo.safetensors", "andrewwe/qwLoras", None),
    ("loras", "flymy_realism.safetensors", "andrewwe/qwLoras", None),
    ("loras", "1GIRL_QWEN_V3.safetensors", "andrewwe/qwLoras", None),
    ("loras", "Edit-R1-Qwen-Image-Edit-2509.safetensors", "andrewwe/qwLoras", None),
    ("loras", "next-scene_lora-v2-3000.safetensors", "andrewwe/qwLoras", None),
    ("loras", "film_still.safetensors", "andrewwe/qwLoras", None),
    ("loras", "detailz_qwen_000024000.safetensors", "andrewwe/qwLoras", None),
    ("loras/Zit", "BystyMega_b8nk.safetensors", "andrewwe/zitLoras", None),
    ("loras/Zit", "MariaBody_000001500.safetensors", "andrewwe/zitLoras", None),
    ("loras/Zit", "b3tternud3s_v2.safetensors", "andrewwe/zitLoras", None),
    ("loras/Zit", "big-nipples.safetensors", "andrewwe/zitLoras", None),
    ("loras/Zit", "bustywoman_bs9ex.safetensors", "andrewwe/zitLoras", None),
    ("loras/Zit", "christina_ch6tina.safetensors", "andrewwe/zitLoras", None),
    ("loras/Zit", "girls_zimage_g5r4l.safetensors", "andrewwe/zitLoras", None),
    ("loras/Zit", "girls_zimage_g5r4l_000001500.safetensors", "andrewwe/zitLoras", None),

    # LoRA-файли з wiikoo/Qwen-lora-nsfw
    ("loras", "[QWEN] Send Nudes Pro - Beta v1.safetensors", "wiikoo/Qwen-lora-nsfw", "loras"),
    ("loras", "reclining_nude_v1_000003500.safetensors", "wiikoo/Qwen-lora-nsfw", "loras"),
    ("loras", "consistence_edit_v2.safetensors", "wiikoo/Qwen-lora-nsfw", "loras2"),
    ("loras", "qwen_snofs.safetensors", "wiikoo/Qwen-lora-nsfw", "loras"),

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
    gpu="L40S",
    volumes={DATA_ROOT: vol},
)
@modal.concurrent(max_inputs=10)
@modal.web_server(8000, startup_timeout=300) # Increased timeout for handling restarts
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

    # Fix detached HEAD and update ComfyUI backend to the latest version
    print("Fixing git branch and updating ComfyUI backend to the latest version...")
    os.chdir(DATA_BASE)
    try:
        # Check if in detached HEAD state
        result = subprocess.run("git symbolic-ref HEAD", shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print("Detected detached HEAD, checking out main branch...")
            subprocess.run("git checkout -B main origin/main", shell=True, check=True, capture_output=True, text=True)
            print("Successfully checked out main branch")
        
        # Configure pull strategy to fast-forward only
        subprocess.run("git config pull.ff only", shell=True, check=True, capture_output=True, text=True)
        
        # Perform git pull
        result = subprocess.run("git pull --ff-only", shell=True, check=True, capture_output=True, text=True)
        print("Git pull output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error updating ComfyUI backend: {e.stderr}")
    except Exception as e:
        print(f"Unexpected error during backend update: {e}")

    # Update ComfyUI-Manager to the latest version
    manager_dir = os.path.join(CUSTOM_NODES_DIR, "ComfyUI-Manager")
    if os.path.exists(manager_dir):
        print("Updating ComfyUI-Manager to the latest version...")
        os.chdir(manager_dir)
        try:
            # Configure pull strategy for ComfyUI-Manager
            subprocess.run("git config pull.ff only", shell=True, check=True, capture_output=True, text=True)
            result = subprocess.run("git pull --ff-only", shell=True, check=True, capture_output=True, text=True)
            print("ComfyUI-Manager git pull output:", result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error updating ComfyUI-Manager: {e.stderr}")
        except Exception as e:
            print(f"Unexpected error during ComfyUI-Manager update: {e}")
        os.chdir(DATA_BASE) # Return to base directory
    else:
        print("ComfyUI-Manager directory not found, installing...")
        try:
            subprocess.run("comfy node install ComfyUI-Manager", shell=True, check=True, capture_output=True, text=True)
            print("ComfyUI-Manager installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"Error installing ComfyUI-Manager: {e.stderr}")

    # Upgrade pip at runtime
    print("Upgrading pip at runtime...")
    try:
        result = subprocess.run("pip install --no-cache-dir --upgrade pip", shell=True, check=True, capture_output=True, text=True)
        print("pip upgrade output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error upgrading pip: {e.stderr}")
    except Exception as e:
        print(f"Unexpected error during pip upgrade: {e}")

    # Upgrade comfy-cli at runtime
    print("Upgrading comfy-cli at runtime...")
    try:
        result = subprocess.run("pip install --no-cache-dir --upgrade comfy-cli", shell=True, check=True, capture_output=True, text=True)
        print("comfy-cli upgrade output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error upgrading comfy-cli: {e.stderr}")
    except Exception as e:
        print(f"Unexpected error during comfy-cli upgrade: {e}")

    # Update ComfyUI frontend by installing requirements
    print("Updating ComfyUI frontend by installing requirements...")
    requirements_path = os.path.join(DATA_BASE, "requirements.txt")
    if os.path.exists(requirements_path):
        try:
            result = subprocess.run(
                f"/usr/local/bin/python -m pip install -r {requirements_path}",
                shell=True,
                check=True,
                capture_output=True,
                text=True
            )
            print("Frontend update output:", result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error updating ComfyUI frontend: {e.stderr}")
        except Exception as e:
            print(f"Unexpected error during frontend update: {e}")
    else:
        print(f"Warning: {requirements_path} not found, skipping frontend update")

    # Configure ComfyUI-Manager: Disable auto-fetch, set weak security, and disable file logging
    manager_config_dir = os.path.join(DATA_BASE, "user", "default", "ComfyUI-Manager")
    manager_config_path = os.path.join(manager_config_dir, "config.ini")
    print("Configuring ComfyUI-Manager: Disabling auto-fetch, setting security_level to weak, and disabling file logging...")
    os.makedirs(manager_config_dir, exist_ok=True)
    config_content = "[default]\nnetwork_mode = private\nsecurity_level = weak\nlog_to_file = false\n"
    with open(manager_config_path, "w") as f:
        f.write(config_content)
    print(f"Updated {manager_config_path} with network_mode=private, security_level=weak, log_to_file=false")

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
