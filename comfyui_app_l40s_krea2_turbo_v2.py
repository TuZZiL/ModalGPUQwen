import hashlib
from concurrent.futures import ThreadPoolExecutor
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
BASE_MODEL_NAME = "krea2_turbo"
APP_NAME = "comfyui-l40s-krea2-turbo-v2"
# Base utility nodes + Krea 2 conditioning rebalance node.
# Krea 2 is a FLUX 2-architecture model loaded via native ComfyUI nodes
# (UNETLoader / CLIPLoader type "krea2" / VAELoader), so no GGUF or
# klein-specific loaders are needed here.
CUSTOM_NODE_REPOS = [
    ("Comfy-Org/ComfyUI-Manager", False),
    ("rgthree/rgthree-comfy", False),
    ("kijai/ComfyUI-KJNodes", True),
    ("TuZZiL/ComfyUI-ConditioningKrea2Rebalance", False),
    ("erosDiffusion/ComfyUI-EulerDiscreteScheduler", False),
    ("capitan01R/ComfyUI-Krea2T-Enhancer", False),
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


def _sync_single_node(repo: str, install_reqs: bool):
    """Sync a single custom node repo (clone or pull + optional pip install)."""
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
            return
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


def sync_custom_node_repos():
    print(f"Synchronizing custom nodes for {BASE_MODEL_NAME}...")
    os.makedirs(CUSTOM_NODES_DIR, exist_ok=True)

    # Parallel git pulls (~3-4s saved vs sequential)
    with ThreadPoolExecutor(max_workers=len(CUSTOM_NODE_REPOS)) as pool:
        pool.map(lambda args: _sync_single_node(*args), CUSTOM_NODE_REPOS)


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


# v2: Heavy workflow template media packages (~430MB) cause 70+ global_subgraph
# requests that block UI loading for minutes. Strip them after requirements install.
STRIP_HEAVY_TEMPLATES = [
    "comfyui-workflow-templates-media-api",
    "comfyui-workflow-templates-media-image",
    "comfyui-workflow-templates-media-other",
    "comfyui-workflow-templates-media-video",
    "comfyui-workflow-templates-media-assets-01",
]


def strip_workflow_template_media():
    """Remove heavy workflow template packages to speed up frontend loading."""
    print("Stripping heavy workflow template media packages...")
    result = subprocess.run(
        ["/usr/local/bin/python", "-m", "pip", "uninstall", "-y"] + STRIP_HEAVY_TEMPLATES,
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print("Stripped workflow template media packages successfully.")
    else:
        print(f"Strip result (non-fatal): {result.stderr.strip()}")


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
    # Libraries required by the custom nodes (kept aligned with the klein9b stack).
    .pip_install("psd-tools", "PyWavelets", "tiktoken", "Wand", "gguf", "diffusers", "peft", "rotary_embedding_torch", "omegaconf", "blake3", "comfy-aimdo", "piexif")
    .run_commands([
        # Bake latest pip/comfy-cli/uv into image to avoid runtime upgrades (~9s saved)
        "pip install --no-cache-dir --upgrade pip comfy-cli uv",
        "pip install --no-cache-dir --force-reinstall --index-url https://download.pytorch.org/whl/cu126 torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0",
        "uv pip install --system --compile-bytecode huggingface_hub[hf_transfer]==0.28.1",
        # Install ComfyUI to default location
        "comfy --skip-prompt install --nvidia"
    ])
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# Bake custom nodes into the image; runtime sync_custom_node_repos keeps them updated.
for repo, install_reqs in CUSTOM_NODE_REPOS:
    image = image.run_commands([git_clone_cmd(repo, install_reqs=install_reqs)])

# Krea 2 Turbo assets.
#   - Model: FP8 (mixed) quant of the FLUX 2-architecture Krea 2 Turbo, ideal for L40S (Ada/RTX 40xx).
#     Load via native "Load Diffusion Model" (UNETLoader) from models/diffusion_models.
#   - Text encoder: Qwen3-VL 4B -> CLIPLoader with type "krea2".
#   - VAE: qwen_image_vae (same VAE family as Anima).
# Turbo inference reference: 8 steps, CFG 0.0, mu 1.15, 1024-2048px.
#
# Model tasks format:
#   1. Using Hugging Face Hub (Preferred/Fastest via hf_transfer):
#      ("subdir", "remote_filename", "repo_id", "subfolder" or None)
#      Example: ("loras/krea2", "fedor_bypass.safetensors", "diobrando0/krea2_loras_public", None)
#      If saving to a different local name, append 5th element:
#      ("subdir", "remote_filename", "repo_id", "subfolder" or None, "local_filename")
#
#   2. Using direct URLs (Fallback for external/custom links starting with http):
#      ("subdir", "local_filename", "https://example.com/file.safetensors", None)
model_tasks = [
    ("diffusion_models", "Krea2_Turbo_fp8mixed.safetensors", "Winnougan/Krea-2-Base-Turbo-NVFP4-FP8-INT8", None),
    ("text_encoders", "qwen3vl_4b_fp8_scaled.safetensors", "Comfy-Org/Qwen3-VL", "text_encoders"),
    ("vae", "qwen_image_vae.safetensors", "Comfy-Org/Qwen-Image_ComfyUI", "split_files/vae"),
    ("vae", "krea2RealVae_v10.safetensors", "andrewwe/kr2", None),
    ("vae", "vae_wan_2.1_vae.safetensors", "EllaPriest45/Krea2_base", None),
    ("loras/krea2", "realism_engine_krea2_v2.safetensors", "Sentinel7/krea2", "2688234/3070702"),
    ("loras/krea2", "MysticXXX_KREA2_v2.safetensors", "Sentinel7/krea2", "2728644/3083062"),
    ("loras/krea2", "krea2_nud3.safetensors", "TechScribe42/krea", "nsfw"),
    ("loras/krea2", "pytorch_lora_weights.safetensors", "Beinsezii/Krea-2-Turbo-Projector-Scale-LoRA-Diffusers", None, "krea2_projector_scale.safetensors"),
    ("loras/krea2", "Krea2-realism-V1.safetensors", "adslkfsajlkj/krea2-realism", "2728365/3066973"),
    ("loras/krea2", "KNPV3_1.safetensors", "Kutches/Kr3a", None),
    ("loras/krea2", "galaxyace_krea2.safetensors", "jjbRs/rs-imagen-models", "loras"),
    ("loras/krea2", "saggy-krea-turbo.safetensors", "Sentinel7/krea2", "1844246/3067822"),
    ("loras/krea2", "BreastSlider-KREA2.safetensors", "Kutches/Kr3a", None),
    ("loras/krea2", "lenovo_krea2.safetensors", "Kutches/Kr3a", None),
    ("loras/krea2", "HMBreasts_krea2_epoch12.safetensors", "Sentinel7/krea2", "2740401/3081828"),
    ("loras/krea2", "Krea2 NSFW+.safetensors", "Sentinel7/krea2", "2742640/3084588", "Krea2_NSFW_plus.safetensors"),
    ("loras/krea2", "krea2_macromastia_clothed.safetensors", "andrewwe/kr2", None),
    ("loras/krea2", "skc3vo.safetensors", "andrewwe/kr2", None),
    ("loras/krea2", "z0jglf.safetensors", "andrewwe/kr2", None),
    ("loras/krea2", "krea2filterbypass3.safetensors", "alienmafio/my-krea2-loras", None),
    ("loras/krea2", "PornMaster Krea2 Detail Slider - Krea2 - -1.5 +1.safetensors", "EllaPriest45/Krea2_actions", None, "PornMaster_Krea2_Detail_Slider.safetensors"),
    ("loras/krea2", "NSFW - Krea2 - Krea2 - Asian,creampie,doggystyle,ebony,hairy,milf,reverse cowgirl,shaved,spreading,suicide girls.safetensors", "EllaPriest45/Krea2_actions", None, "NSFW_Krea2_actions.safetensors"),
    ("loras/krea2", "snofs_krea_v1.safetensors", "alienmafio/my-krea2-loras", None),
    ("loras/krea2", "KNPV4.1_pre.safetensors", "Kutches/Kr3a", None),
    ("loras/krea2", "fedor_bypass.safetensors", "diobrando0/krea2_loras_public", None),
    ("loras/krea2", "refiner_neuter_patch.safetensors", "Hippotes/Krea-2-Experiments", None),
    ("loras/krea2", "Krea2-realism-V2.safetensors", "andrewwe/kr2", None),
    ("loras/krea2", "HMCum_krea2_epoch30.safetensors", "andrewwe/kr2", None),
    ("loras/krea2", "RealisticSnapshotKrea2.safetensors", "andrewwe/kr2", None),
    ("loras/krea2", "impreal.safetensors", "andrewwe/kr2", None),
    ("loras/krea2", "krea2_realism_lora.safetensors", "bonticario/Krea-2-Realism-LoRA", None),
]

# Create volume (dedicated to the Krea 2 stack to keep it isolated from the klein9b volume)
vol = modal.Volume.from_name("comfyui-krea2", create_if_missing=True)

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
    # v2: removed upgrade_runtime_tools_author_style() — pip/comfy-cli baked in image (~9s saved)
    # v2: replaced update_comfyui_frontend with hash-based sync (~23s saved)
    sync_frontend_requirements(os.path.join(DATA_BASE, "requirements.txt"))
    strip_workflow_template_media()
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

    # Ensure all required directories exist for the Krea 2 Turbo stack
    required_dirs = [
        CUSTOM_NODES_DIR,
        MODELS_DIR,
        os.path.join(MODELS_DIR, "diffusion_models"),
        os.path.join(MODELS_DIR, "text_encoders"),
        os.path.join(MODELS_DIR, "vae"),
        os.path.join(MODELS_DIR, "loras"),
        os.path.join(MODELS_DIR, "loras", "krea2"),
        TMP_DL,
    ]

    for d in required_dirs:
        os.makedirs(d, exist_ok=True)

    # Download Krea 2 Turbo models at runtime (only if missing)
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

    # v2: pinned frontend version to avoid GitHub HTTP lookup (~1-2s saved)
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
        "Comfy-Org/ComfyUI_frontend@1.47.6",
    ]
    print(f"Executing: {' '.join(cmd)}")

    subprocess.Popen(
        cmd,
        cwd=DATA_BASE,
        env=os.environ.copy()
    )
