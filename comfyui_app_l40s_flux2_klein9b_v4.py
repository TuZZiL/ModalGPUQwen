import hashlib
import os
import shutil
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from typing import Optional
from huggingface_hub import hf_hub_download
import modal

# Paths
DATA_ROOT = "/data/comfy"
DATA_BASE = os.path.join(DATA_ROOT, "ComfyUI")
CUSTOM_NODES_DIR = os.path.join(DATA_BASE, "custom_nodes")
MODELS_DIR = os.path.join(DATA_BASE, "models")
TMP_DL = "/tmp/download"
RUNTIME_STATE_DIR = os.path.join(DATA_ROOT, ".runtime_state")
FRONTEND_REQUIREMENTS_HASH = os.path.join(RUNTIME_STATE_DIR, "requirements.sha256")
GPU_TYPE = "L40S"
BASE_MODEL_NAME = "flux2_klein9b"
APP_NAME = "comfyui-l40s-flux2-klein9b"
CUSTOM_NODE_REPOS = [
    ("city96/ComfyUI-GGUF", True),
    ("rgthree/rgthree-comfy", False),
    ("kijai/ComfyUI-KJNodes", True),
    ("TuZZiL/tuz-fluxklein-toolkit", True),
    ("ClownsharkBatwing/RES4LYF", True),
]

# ComfyUI default install location
DEFAULT_COMFY_DIR = "/root/comfy/ComfyUI"

def git_clone_cmd(node_repo: str, recursive: bool = False, install_reqs: bool = False) -> str:
    name = node_repo.split("/")[-1]
    dest = os.path.join(DEFAULT_COMFY_DIR, "custom_nodes", name)
    cmd = "git clone"
    if recursive:
        cmd += " --recursive"
    cmd += f" https://github.com/{node_repo} {dest}"
    if install_reqs:
        cmd += f" && if [ -f {dest}/requirements.txt ]; then pip install -r {dest}/requirements.txt; fi"
    return cmd


def run_shell(command: str, cwd: Optional[str] = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        shell=True,
        check=check,
        capture_output=True,
        text=True,
        cwd=cwd,
    )


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_comfyui_on_volume():
    if os.path.exists(os.path.join(DATA_BASE, "main.py")):
        return

    print("First run detected. Copying ComfyUI from default location to volume...")
    os.makedirs(DATA_ROOT, exist_ok=True)

    if os.path.exists(DEFAULT_COMFY_DIR):
        print(f"Copying {DEFAULT_COMFY_DIR} to {DATA_BASE}")
        shutil.copytree(DEFAULT_COMFY_DIR, DATA_BASE, dirs_exist_ok=True)
        return

    print(f"Warning: {DEFAULT_COMFY_DIR} not found, creating empty structure")
    os.makedirs(DATA_BASE, exist_ok=True)


def update_comfyui_backend_author_style():
    print("Updating ComfyUI backend to the latest version...")
    os.chdir(DATA_BASE)
    try:
        result = subprocess.run("git symbolic-ref HEAD", shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print("Detected detached HEAD, fetching and checking out main branch...")
            subprocess.run("git fetch --all", shell=True, check=True, capture_output=True, text=True)
            subprocess.run("git checkout -B main origin/main", shell=True, check=True, capture_output=True, text=True)
            print("Successfully checked out main branch")

        subprocess.run("git config pull.ff only", shell=True, check=True, capture_output=True, text=True)
        result = subprocess.run("git pull --ff-only", shell=True, check=True, capture_output=True, text=True)
        print("Git pull output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error updating ComfyUI backend: {e.stderr}")
    except Exception as e:
        print(f"Unexpected error during backend update: {e}")


def update_comfyui_manager_author_style():
    manager_dir = os.path.join(CUSTOM_NODES_DIR, "ComfyUI-Manager")
    if os.path.exists(manager_dir):
        print("Updating ComfyUI-Manager to the latest version...")
        os.chdir(manager_dir)
        try:
            subprocess.run("git config pull.ff only", shell=True, check=True, capture_output=True, text=True)
            result = subprocess.run("git pull --ff-only", shell=True, check=True, capture_output=True, text=True)
            print("ComfyUI-Manager git pull output:", result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error updating ComfyUI-Manager: {e.stderr}")
        except Exception as e:
            print(f"Unexpected error during ComfyUI-Manager update: {e}")
        os.chdir(DATA_BASE)
    else:
        print("ComfyUI-Manager directory not found, installing...")
        try:
            subprocess.run("comfy node install ComfyUI-Manager", shell=True, check=True, capture_output=True, text=True)
            print("ComfyUI-Manager installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"Error installing ComfyUI-Manager: {e.stderr}")


def update_comfyui_frontend_author_style():
    print("Updating ComfyUI frontend by installing requirements...")
    requirements_path = os.path.join(DATA_BASE, "requirements.txt")
    if os.path.exists(requirements_path):
        try:
            result = subprocess.run(
                f"/usr/local/bin/python -m pip install -r {requirements_path}",
                shell=True,
                check=True,
                capture_output=True,
                text=True,
            )
            print("Frontend update output:", result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error updating ComfyUI frontend: {e.stderr}")
        except Exception as e:
            print(f"Unexpected error during frontend update: {e}")
    else:
        print(f"Warning: {requirements_path} not found, skipping frontend update")


def upgrade_runtime_tools_author_style():
    print("Upgrading pip at runtime...")
    try:
        result = subprocess.run("pip install --upgrade pip", shell=True, check=True, capture_output=True, text=True)
        print("pip upgrade output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error upgrading pip: {e.stderr}")
    except Exception as e:
        print(f"Unexpected error during pip upgrade: {e}")

    print("Upgrading comfy-cli at runtime...")
    try:
        result = subprocess.run("pip install --no-cache-dir --upgrade comfy-cli", shell=True, check=True, capture_output=True, text=True)
        print("comfy-cli upgrade output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error upgrading comfy-cli: {e.stderr}")
    except Exception as e:
        print(f"Unexpected error during comfy-cli upgrade: {e}")


def configure_comfyui_manager_author_style():
    config_content = "[default]\nnetwork_mode = private\nsecurity_level = weak\nlog_to_file = false\n"
    config_paths = [
        os.path.join(DATA_BASE, "user", "__manager", "config.ini"),
        os.path.join(DATA_BASE, "user", "default", "ComfyUI-Manager", "config.ini"),
    ]

    print("Configuring ComfyUI-Manager: Disabling auto-fetch, setting security_level to weak, and disabling file logging...")
    for config_path in config_paths:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as handle:
            handle.write(config_content)
        print(f"Updated {config_path} with network_mode=private, security_level=weak, log_to_file=false")


def detect_remote_branch(repo_dir: str) -> Optional[str]:
    run_shell("git remote set-head origin -a", cwd=repo_dir, check=False)

    origin_head = run_shell(
        "git symbolic-ref --short refs/remotes/origin/HEAD",
        cwd=repo_dir,
        check=False,
    )
    if origin_head.returncode == 0:
        remote_ref = origin_head.stdout.strip()
        if remote_ref.startswith("origin/"):
            return remote_ref.split("/", 1)[1]

    for candidate in ("main", "master"):
        probe = run_shell(
            f"git show-ref --verify --quiet refs/remotes/origin/{candidate}",
            cwd=repo_dir,
            check=False,
        )
        if probe.returncode == 0:
            return candidate

    return None


def update_git_repo(repo_dir: str, label: str):
    repo_probe = run_shell("git rev-parse --is-inside-work-tree", cwd=repo_dir, check=False)
    if repo_probe.returncode != 0:
        print(f"Skipping {label} update: {repo_dir} is not a git repository.")
        return

    branch = detect_remote_branch(repo_dir)
    if not branch:
        print(f"Skipping {label} update: could not determine remote branch for origin.")
        return

    head_probe = run_shell("git symbolic-ref --short HEAD", cwd=repo_dir, check=False)
    if head_probe.returncode != 0:
        print(f"Detected detached HEAD in {label}, checking out origin/{branch}...")
        checkout = run_shell(
            f"git checkout -B {branch} origin/{branch}",
            cwd=repo_dir,
            check=False,
        )
        if checkout.returncode != 0:
            details = checkout.stderr.strip() or checkout.stdout.strip()
            print(f"Skipping {label} update: failed to checkout origin/{branch}: {details}")
            return

    run_shell("git config pull.ff only", cwd=repo_dir, check=False)
    pull = run_shell(f"git pull --ff-only origin {branch}", cwd=repo_dir, check=False)
    if pull.returncode == 0:
        output = pull.stdout.strip() or "Already up to date."
        print(f"{label} git pull output: {output}")
        return

    details = pull.stderr.strip() or pull.stdout.strip()
    print(f"Error updating {label}: {details}")


def sync_custom_node_repos():
    print(f"Synchronizing custom nodes for {BASE_MODEL_NAME}...")
    os.makedirs(CUSTOM_NODES_DIR, exist_ok=True)

    for repo, install_reqs in CUSTOM_NODE_REPOS:
        repo_name = repo.split("/")[-1]
        repo_dir = os.path.join(CUSTOM_NODES_DIR, repo_name)
        label = f"custom node {repo_name}"

        if not os.path.exists(os.path.join(repo_dir, ".git")):
            print(f"Cloning {label}...")
            clone = run_shell(
                f"git clone https://github.com/{repo} {repo_dir}",
                cwd=CUSTOM_NODES_DIR,
                check=False,
            )
            if clone.returncode != 0:
                details = clone.stderr.strip() or clone.stdout.strip()
                print(f"Error cloning {label}: {details}")
                continue
        else:
            update_git_repo(repo_dir, label)

        if install_reqs:
            requirements_path = os.path.join(repo_dir, "requirements.txt")
            if os.path.exists(requirements_path):
                try:
                    result = subprocess.run(
                        ["/usr/local/bin/python", "-m", "pip", "install", "-r", requirements_path],
                        check=True,
                        capture_output=True,
                        text=True,
                        cwd=repo_dir,
                    )
                    print(f"{label} requirements output:", result.stdout)
                except subprocess.CalledProcessError as e:
                    print(f"Error installing requirements for {label}: {e.stderr}")


def sync_frontend_requirements(requirements_path: str):
    if not os.path.exists(requirements_path):
        print(f"Warning: {requirements_path} not found, skipping frontend update")
        return

    os.makedirs(RUNTIME_STATE_DIR, exist_ok=True)
    current_hash = file_sha256(requirements_path)
    previous_hash = None
    if os.path.exists(FRONTEND_REQUIREMENTS_HASH):
        with open(FRONTEND_REQUIREMENTS_HASH, "r", encoding="utf-8") as handle:
            previous_hash = handle.read().strip()

    if previous_hash == current_hash:
        print("ComfyUI frontend requirements already match the current requirements.txt, skipping install.")
        return

    print("Installing ComfyUI frontend requirements because requirements.txt changed...")
    result = subprocess.run(
        ["/usr/local/bin/python", "-m", "pip", "install", "-r", requirements_path],
        check=True,
        capture_output=True,
        text=True,
        cwd=DATA_BASE,
    )
    print("Frontend update output:", result.stdout)

    with open(FRONTEND_REQUIREMENTS_HASH, "w", encoding="utf-8") as handle:
        handle.write(current_hash)


def probe_runtime_dependencies():
    print(f"Runtime python: {sys.executable}")
    for package_name in ("blake3", "comfy-aimdo", "torch", "torchvision", "torchaudio"):
        try:
            print(f"Runtime package: {package_name}={version(package_name)}")
        except PackageNotFoundError:
            print(f"Runtime package: {package_name}=MISSING")

def download_model(subdir: str, filename: str, primary_source: dict, backup_source: Optional[dict] = None, local_filename: Optional[str] = None):
    target_dir = os.path.join(MODELS_DIR, subdir)
    os.makedirs(target_dir, exist_ok=True)
    target_name = local_filename if local_filename else filename
    target_path = os.path.join(target_dir, target_name)

    if os.path.exists(target_path):
        print(f"Model {target_name} already exists, skipping download.")
        return

    sources = [primary_source]
    if backup_source:
        sources.append(backup_source)

    for i, source in enumerate(sources):
        source_type = "Backup" if i > 0 else "Primary"
        print(f"Attempting {source_type} download for {target_name}...")
        try:
            source_url = source.get("url")
            repo = source.get("repo_id")
            subf = source.get("subfolder")
            if source_url or (isinstance(repo, str) and repo.startswith("http")):
                # Direct download from a resolved file URL.
                download_url = source_url or repo
                print(f"Downloading from URL: {download_url}")
                subprocess.run(["wget", "-O", os.path.join(TMP_DL, filename), download_url], check=True)
                shutil.move(f"{TMP_DL}/{filename}", target_path)
            else:
                # HF download
                print(f"Downloading from HF: {repo}/{subf if subf else ''}")
                out = hf_hub_download(repo_id=repo, filename=filename, subfolder=subf, local_dir=TMP_DL)
                shutil.move(out, target_path)
            
            print(f"Successfully downloaded {target_name} from {source_type} source.")
            return
        except Exception as e:
            print(f"Failed to download from {source_type} source: {e}")
            if i == len(sources) - 1:
                print(f"All sources failed for {target_name}")
            else:
                print("Trying next source...")

# Build image with ComfyUI installed to default location /root/comfy/ComfyUI
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "wget", "libgl1-mesa-glx", "libglib2.0-0", "ffmpeg", "imagemagick", "libmagickwand-dev")
    # 👇 ВИПРАВЛЕННЯ: Додано необхідні бібліотеки для кастомних нод
    .pip_install("psd-tools", "PyWavelets", "tiktoken", "Wand", "gguf", "diffusers", "peft", "rotary_embedding_torch", "omegaconf", "blake3", "comfy-aimdo", "piexif")
    .run_commands([
        "pip install --upgrade pip",
        "pip install --no-cache-dir --force-reinstall --index-url https://download.pytorch.org/whl/cu126 torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0",
        "pip install --no-cache-dir comfy-cli uv",
        "uv pip install --system --compile-bytecode huggingface_hub[hf_transfer]==0.28.1",
        # Install ComfyUI to default location
        "comfy --skip-prompt install --nvidia"
    ])
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# Custom nodes synchronized with QuickPod `quick_download_quickpod_codex_v2.sh`
for repo, install_reqs in CUSTOM_NODE_REPOS:
    image = image.run_commands([git_clone_cmd(repo, install_reqs=install_reqs)])

# FLUX 2 Klein 9B assets synchronized with QuickPod `quick_download_quickpod_codex_v2.sh`
model_tasks = [
    ("unet/FLUX", "flux-2-klein-9b-Q8_0.gguf", "unsloth/FLUX.2-klein-9B-GGUF", None),
    ("text_encoders", "qwen_3_8b_fp8mixed.safetensors", "Comfy-Org/vae-text-encorder-for-flux-klein-9b", "split_files/text_encoders"),
    ("text_encoders", "Qwen3-8B-Gemini-2.5-Flash-Uncensored-Q8_0.gguf", "wazimondo/Qwen3-Uncensored-TextEncoders-FLUX-Klein-Z-Image-Turbo-GGUF", None),
    ("vae", "flux2-vae.safetensors", "Comfy-Org/vae-text-encorder-for-flux-klein-9b", "split_files/vae"),
    ("loras/FLUX9bKlein", "The_Body_Version_A_Flux2.k.9B_r16_AdamW8Bit_Weighted_768_woman_000005000.safetensors", "andrewwe/klein9bl", None),
    ("loras/FLUX9bKlein", "The_Body_Version_M_Flux.2.klein.9B.r16._000005000.safetensors", "andrewwe/klein9bl", None),
    ("loras/FLUX9bKlein", "detail_slider_klein_9b_20260123_065513.safetensors", "andrewwe/klein9bl", None),
    ("loras/FLUX9bKlein", "f2_klein9b_macromastia_clothed.safetensors", "andrewwe/klein9bl", None),
    ("loras/FLUX9bKlein", "klein_slider_anatomy.safetensors", "andrewwe/klein9bl", None),
    ("loras/FLUX9bKlein", "nipplediffusion-f2-klein-9b_v3.safetensors", "Sentinel7/flux2", "2331032/2749020"),
    ("loras/FLUX9bKlein", "NSFW-klein.safetensors", "andrewwe/klein9bl", None),
    ("loras/FLUX9bKlein", "f2_klein9b_macromastia_naked.safetensors", "andrewwe/klein9bl", None),
    ("loras/FLUX9bKlein", "Flux_Klein_9B_Nude_V1_000000750.safetensors", "andrewwe/klein9bl", None),
    ("loras/FLUX9bKlein", "Flux2Klein9BCumAnywhere.safetensors", "andrewwe/klein9bl", None),
    ("loras/FLUX9bKlein", "breast_slider_klein9b_v09_20260202_070616.safetensors", "UnifiedHorusRA/TheFourHorsemen", "The_Breast_Slider_-_Klein_Edition/Flux_2_Klein_9B"),
    ("loras/FLUX9bKlein", "Klein9BGeneralPenis-v1-0.safetensors", "UnifiedHorusRA/TheFourHorsemen", "Klein_9B_General_Penis_Lora/Flux_2_Klein_9B"),
    ("loras/FLUX9bKlein", "Penis_edit_V01.safetensors", "UnifiedHorusRA/TheFourHorsemen", "erect_penis_Flux_2_Klein_9B/Flux_2_Klein_9B"),
    ("loras/FLUX9bKlein", "klein-deepthroat-15epoc-k3nk.safetensors", "andrewwe/klein9bl", None),
    ("loras/FLUX9bKlein", "bj_20260120_22-22-29epoch15_comfy.safetensors", "UnifiedHorusRA/TheFourHorsemen", "flux2-klein-9b_Pyros_BJ/Flux_2_Klein_9B"),
    ("loras/FLUX9bKlein", "eros_fklein_v2_000019795.safetensors", "andrewwe/klein9bl", None),
    ("loras/FLUX9bKlein", "clothesonoffv2.safetensors", "Sentinel7/flux2", "2337249/2665761"),
    ("loras/FLUX9bKlein", "removedress3000steps4_3.safetensors", "andrewwe/klein9bl", None),
    ("loras/FLUX9bKlein", "RemoveDressKlein9b_3.safetensors", "andrewwe/klein9bl", None),
    ("loras/FLUX9bKlein", "RemoveDressK9B_v2_6.safetensors", "andrewwe/klein9bl", None),
    ("loras/FLUX9bKlein", "removedress5000_5.safetensors", "andrewwe/klein9bl", None),
    ("loras/FLUX9bKlein", "removedress4000steps5_4.safetensors", "andrewwe/klein9bl", None),
    ("loras/FLUX9bKlein", "remove_dressv3k_2.safetensors", "andrewwe/klein9bl", None),
    ("loras/FLUX9bKlein", "nipplediffusion-saggy-f2-klein-9b_v1.safetensors", "andrewwe/klein9bl", None),
    ("loras/FLUX9bKlein", "nipplediffusionFlatF2.hdvQ.safetensors", "andrewwe/klein9bl", None),
    ("loras/FLUX9bKlein", "f2k_consist_20260225.safetensors", "lrzjason/Consistance_Edit_Lora", None),
    ("loras/FLUX9bKlein", "PornMaster_flat_chest_flux-2-klein-9b_V1_B.safetensors", "andrewwe/klein9bl", None),
    ("loras/FLUX9bKlein", "removedress6000_6.safetensors", "andrewwe/klein9bl", None),
    ("loras/FLUX9bKlein", "hairy-klein.safetensors", "andrewwe/klein9bl", None),
    ("loras/FLUX9bKlein", "removedress4000fullprompt_4.safetensors", "andrewwe/klein9bl", None),
    ("loras/FLUX9bKlein", "ChrisHendriks_v3_3.safetensors", "andrewwe/klein9bl", None),
    ("loras/FLUX9bKlein", "Klein-consistency.safetensors", "dx8152/Flux2-Klein-9B-Consistency", None),
    ("loras/FLUX9bKlein", "klein_9b_enhancer_v2.safetensors", "reverentelusarca/detail-enhancer-flux-klein-9b", None),
    ("loras/FLUX9bKlein", "realistic.safetensors", "joseph0017/Flux2-Klein-9B-Enhanced-Details", None),
    ("loras/FLUX9bKlein", "f2k_9B_lcs_consist_preview.safetensors", "https://huggingface.co/Sentinel7/flux2/resolve/main/1939453/2810265/f2k_9B_lcs_consist_preview.safetensors", None),
    ("loras/FLUX9bKlein", "Remove_Clothing_Censor.safetensors", "https://huggingface.co/Sentinel7/flux2/resolve/main/730405/2850798/Remove_Clothing_Censor.safetensors", None),
    ("loras/FLUX9bKlein", "Leaked%20nudes%20v3.safetensors", "https://huggingface.co/creatormirai/AssumethePosition/resolve/main/Klein/Leaked%20nudes%20v3.safetensors", None, "Leaked nudes v3.safetensors"),
    ("loras/FLUX9bKlein", "cum_on_face_v2.safetensors", "https://huggingface.co/creatormirai/AssumethePosition/resolve/main/Klein/cum_on_face_v2.safetensors", None),
    ("loras/FLUX9bKlein", "HighResolution9B.safetensors", "andrewwe/klein9bl", None),
    ("loras/FLUX9bKlein", "BustyWomens_v1_7.safetensors", "andrewwe/klein9bl", None),
    ("loras/FLUX9bKlein", "SEXGOD_ImprovedNudity_Klein9b_v3.safetensors", "andrewwe/klein9bl", None),
    ("loras/FLUX9bKlein", "Flux2-Klein-9B-consistency-V2.safetensors", "dx8152/Flux2-Klein-9B-Consistency", None),
    ("loras/FLUX9bKlein", "klein_snofs_v1_3.safetensors", "sintecs/flux_klein_loras", None),
    ("loras/FLUX9bKlein", "remove_dress1904_4.safetensors", "andrewwe/klein9bl", None),
    ("loras/FLUX9bKlein", "pussydiffusion-f2-klein-9b_v2.safetensors", "https://huggingface.co/UnifiedHorusRA/TheFourHorsemen/resolve/main/PussyDiffusion_-_Flux2_Klein/Flux_2_Klein_9B/pussydiffusion-f2-klein-9b_v2.safetensors", None),
    ("loras/FLUX9bKlein", "Realism_Engine_Klein_V1.safetensors", "https://huggingface.co/UnifiedHorusRA/TheFourHorsemen/resolve/main/Realism_Engine_Klein/Flux_2_Klein_9B/Realism_Engine_Klein_V1.safetensors", None),
    ("loras/FLUX9bKlein", "klein-m4crom4sti4-v2-3epoc-k3nk.safetensors", "Sentinel7/flux2", "2327401/2710450"),
    ("loras/FLUX9bKlein", "Chest_9Bslider.safetensors", "andrewwe/klein9bl", None),
    ("loras/FLUX9bKlein", "buttocksslider.safetensors", "andrewwe/klein9bl", None),
    ("loras/FLUX9bKlein", "SheerSeeThrough_F2K9B_v1.safetensors", "andrewwe/klein9bl", None),
    ("loras/FLUX9bKlein", "Remove_DressMax_9.safetensors", "andrewwe/klein9bl", None),
    ("loras/FLUX9bKlein", "Remove_DressMax_8.safetensors", "andrewwe/klein9bl", None),
    ("loras/FLUX9bKlein", "Remove_DressMax_7.safetensors", "andrewwe/klein9bl", None),
    ("loras/FLUX9bKlein", "Flat_C_2.safetensors", "andrewwe/klein9bl", None),
    ("loras/FLUX9bKlein", "Flat_C_6.safetensors", "andrewwe/klein9bl", None),
    ("loras/FLUX9bKlein", "Flat_C_8.safetensors", "andrewwe/klein9bl", None),
    ("loras/FLUX9bKlein", "Flat_C_10.safetensors", "andrewwe/klein9bl", None),
    ("loras/FLUX9bKlein", "HugeDick_c1-st3000.safetensors", "andrewwe/klein9bl", None),
    ("loras/FLUX9bKlein", "HugeDick_c1-st4000.safetensors", "andrewwe/klein9bl", None),
    ("loras/FLUX9bKlein", "TribeW_v1_c1-st3000.safetensors", "andrewwe/klein9bl", None),
    ("loras/FLUX9bKlein", "TribeW_v1_c1-st4000.safetensors", "andrewwe/klein9bl", None),
    ("loras/FLUX9bKlein", "TribeW_v1_c1-st5000.safetensors", "andrewwe/klein9bl", None),
    ("loras/FLUX9bKlein", "TribeW_v1_c1-st6000.safetensors", "andrewwe/klein9bl", None),
    ("loras/FLUX9bKlein", "SexGod_Klein9b_ImageEdit_NudityHelper_v1.safetensors", "SexGod1979/Flux.2_Klein9b_ImageEdit_NudityHelper", None),
    ("loras/FLUX9bKlein", "Klein2-9B-SmartCharacterSwap.safetensors", "nhathoangfoto/Flux.2-Klein-9B-SmartCharacterSwap", None),
    ("loras/FLUX9bKlein", "NaturalBeautyFLUX2Klein9BNudity_v2.safetensors", "codeShare/flux-klein-9B-loras", None),
    ("loras/FLUX9bKlein", "34O CUP M4CROM4STI4 v2.0 - Klein9B - large-gigantic pendulous breasts.safetensors", "EllaPriest45/Klein9B_Actions", None),
    ("loras/FLUX9bKlein", "C.H.E.S.T. Show - Klein9B.safetensors", "EllaPriest45/Klein9B_Actions", None),
    ("loras/FLUX9bKlein", "Pussy FIX - Klein9B - pusfix,pubic area,genital area.safetensors", "EllaPriest45/Klein9B_Actions", None),
]

# Create volume
vol = modal.Volume.from_name("comfyui-app", create_if_missing=True)

app = modal.App(name=APP_NAME, image=image)

@app.function(
    max_containers=1,
    scaledown_window=300,
    timeout=7200,
    gpu=GPU_TYPE,
    volumes={DATA_ROOT: vol},
)
@modal.concurrent(max_inputs=10)
@modal.web_server(8000, startup_timeout=1800)
def ui():
    ensure_comfyui_on_volume()

    update_comfyui_backend_author_style()
    upgrade_runtime_tools_author_style()
    update_comfyui_frontend_author_style()
    update_comfyui_manager_author_style()
    configure_comfyui_manager_author_style()

    try:
        sync_custom_node_repos()
    except Exception as e:
        print(f"Unexpected error during custom node sync: {e}")

    print("Probing runtime dependencies before launching ComfyUI...")
    try:
        probe_runtime_dependencies()
    except Exception as e:
        print(f"Runtime dependency probe failed: {e}")
        raise
    print("Runtime dependency probe passed.")

    # Ensure all required directories exist for the FLUX 2 Klein 9B stack
    required_dirs = [
        CUSTOM_NODES_DIR,
        MODELS_DIR,
        os.path.join(MODELS_DIR, "unet"),
        os.path.join(MODELS_DIR, "unet", "FLUX"),
        os.path.join(MODELS_DIR, "text_encoders"),
        os.path.join(MODELS_DIR, "vae"),
        os.path.join(MODELS_DIR, "loras"),
        os.path.join(MODELS_DIR, "loras", "FLUX9bKlein"),
        TMP_DL,
    ]
    
    for d in required_dirs:
        os.makedirs(d, exist_ok=True)

    # Download FLUX 2 Klein 9B models at runtime (only if missing)
    print(f"Checking and downloading missing {BASE_MODEL_NAME} models...")
    for task in model_tasks:
        sub, fn, repo, subf = task[:4]
        local_fn = task[4] if len(task) > 4 else None

        display_name = local_fn if local_fn else fn
        target = os.path.join(MODELS_DIR, sub, display_name)

        if not os.path.exists(target):
            print(f"Downloading {fn} as {display_name} to {target}...")
            primary = {"repo_id": repo, "subfolder": subf}
            download_model(sub, fn, primary, local_filename=local_fn)
        else:
            print(f"Model {display_name} already exists, skipping download")

    # Set COMFY_DIR environment variable to volume location
    os.environ["COMFY_DIR"] = DATA_BASE

    # Launch ComfyUI from volume location
    print(f"Starting ComfyUI from {DATA_BASE} on {GPU_TYPE} with {BASE_MODEL_NAME} support...")
    
    # Start ComfyUI server with correct syntax and latest frontend
    cmd = [
        "comfy",
        "launch",
        "--",
        "--listen",
        "0.0.0.0",
        "--port",
        "8000",
        "--enable-cors-header",
        "--front-end-version",
        "Comfy-Org/ComfyUI_frontend@latest",
    ]
    print(f"Executing: {' '.join(cmd)}")
    
    subprocess.Popen(
        cmd,
        cwd=DATA_BASE,
        env=os.environ.copy()
    )
