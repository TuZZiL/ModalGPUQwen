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
    ("TuZZiL/Comfyui-flux2klein-Lora-loader", True),
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
            if "url" in source:
                # Direct download (Civitai, etc.)
                print(f"Downloading from URL: {source['url']}")
                # Use wget for direct URLs
                subprocess.run(f"wget -O {TMP_DL}/{filename} \"{source['url']}\"", shell=True, check=True)
                shutil.move(f"{TMP_DL}/{filename}", target_path)
            else:
                # HF download
                repo = source.get("repo_id")
                subf = source.get("subfolder")
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
    .pip_install("psd-tools", "PyWavelets", "tiktoken", "Wand", "gguf", "diffusers", "peft", "rotary_embedding_torch", "omegaconf", "blake3", "comfy-aimdo")
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
    ("vae", "flux2-vae.safetensors", "Comfy-Org/vae-text-encorder-for-flux-klein-9b", "split_files/vae"),
    ("loras/FLUX9bKlein", "The_Body_Version_A_Flux2.k.9B_r16_AdamW8Bit_Weighted_768_woman_000005000.safetensors", "andrewwe/klein9bl", None),
    ("loras/FLUX9bKlein", "The_Body_Version_M_Flux.2.klein.9B.r16._000005000.safetensors", "andrewwe/klein9bl", None),
    ("loras/FLUX9bKlein", "detail_slider_klein_9b_20260123_065513.safetensors", "andrewwe/klein9bl", None),
    ("loras/FLUX9bKlein", "f2_klein9b_macromastia_clothed.safetensors", "andrewwe/klein9bl", None),
    ("loras/FLUX9bKlein", "klein_slider_anatomy.safetensors", "andrewwe/klein9bl", None),
    ("loras/FLUX9bKlein", "nipplediffusion-f2-klein-9b_v3.safetensors", "Sentinel7/flux2", "2331032/2749020"),
    ("loras/FLUX9bKlein", "NSFW-klein.safetensors", "andrewwe/klein9bl", None),
    ("loras/FLUX9bKlein", "f2_klein9b_macromastia_naked.safetensors", "andrewwe/klein9bl", None),
    ("loras/FLUX9bKlein", "pussydiffusion-f2-klein-9b.safetensors", "andrewwe/klein9bl", None),
    ("loras/FLUX9bKlein", "Flux_Klein_9B_Nude_V1_000000750.safetensors", "andrewwe/klein9bl", None),
    ("loras/FLUX9bKlein", "klein_snofs_v1_1.safetensors", "Sentinel7/flux2", "1972981/2695876"),
    ("loras/FLUX9bKlein", "Flux2Klein9BCumAnywhere.safetensors", "andrewwe/klein9bl", None),
    ("loras/FLUX9bKlein", "SEXGOD_ImprovedNudity_Klein9b_v2_5.safetensors", "Sentinel7/flux2", "2439952/2755254"),
    ("loras/FLUX9bKlein", "breast_slider_klein9b_v09_20260202_070616.safetensors", "UnifiedHorusRA/TheFourHorsemen", "The_Breast_Slider_-_Klein_Edition/Flux_2_Klein_9B"),
    ("loras/FLUX9bKlein", "Klein9BGeneralPenis-v1-0.safetensors", "UnifiedHorusRA/TheFourHorsemen", "Klein_9B_General_Penis_Lora/Flux_2_Klein_9B"),
    ("loras/FLUX9bKlein", "Penis_edit_V01.safetensors", "UnifiedHorusRA/TheFourHorsemen", "erect_penis_Flux_2_Klein_9B/Flux_2_Klein_9B"),
    ("loras/FLUX9bKlein", "klein-deepthroat-15epoc-k3nk.safetensors", "andrewwe/klein9bl", None),
    ("loras/FLUX9bKlein", "bj_20260120_22-22-29epoch15_comfy.safetensors", "UnifiedHorusRA/TheFourHorsemen", "flux2-klein-9b_Pyros_BJ/Flux_2_Klein_9B"),
    ("loras/FLUX9bKlein", "eros_fklein_v1_000017100.safetensors", "andrewwe/klein9bl", None),
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
    ("loras/FLUX9bKlein", "ChrisHendricks_v2_4.safetensors", "andrewwe/klein9bl", None),
    ("loras/FLUX9bKlein", "ChrisHendriks_v3_3.safetensors", "andrewwe/klein9bl", None),
    ("loras/FLUX9bKlein", "Klein-consistency.safetensors", "dx8152/Flux2-Klein-9B-Consistency", None),
    ("loras/FLUX9bKlein", "klein_9b_enhancer_v2.safetensors", "reverentelusarca/detail-enhancer-flux-klein-9b", None),
    ("loras/FLUX9bKlein", "realistic.safetensors", "joseph0017/Flux2-Klein-9B-Enhanced-Details", None),
    ("loras/FLUX9bKlein", "BustyWomens_v1_7.safetensors", "andrewwe/klein9bl", None),
    ("loras/FLUX9bKlein", "SEXGOD_ImprovedNudity_Klein9b_v3.safetensors", "andrewwe/klein9bl", None),
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

    # Diagnostic temporary fix: keep the ComfyUI backend pinned to the volume state.
    print("Skipping ComfyUI backend update for this diagnostic run.")

    print("Skipping runtime pip/comfy-cli upgrades; these should come from the built image.")

    # Update ComfyUI frontend by installing requirements
    print("Updating ComfyUI frontend by installing requirements...")
    requirements_path = os.path.join(DATA_BASE, "requirements.txt")
    try:
        sync_frontend_requirements(requirements_path)
    except subprocess.CalledProcessError as e:
        print(f"Error updating ComfyUI frontend: {e.stderr}")
    except Exception as e:
        print(f"Unexpected error during frontend update: {e}")

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
    cmd = ["comfy", "launch", "--", "--listen", "0.0.0.0", "--port", "8000"]
    print(f"Executing: {' '.join(cmd)}")
    
    subprocess.Popen(
        cmd,
        cwd=DATA_BASE,
        env=os.environ.copy()
    )
