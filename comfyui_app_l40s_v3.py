import hashlib
import os
import shutil
import subprocess
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
    .pip_install("psd-tools", "PyWavelets", "tiktoken", "Wand", "gguf", "diffusers", "peft", "rotary_embedding_torch", "omegaconf")
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
    ("Fannovel16/ComfyUI-MagickWand", {'install_reqs': True}),
    ("numz/ComfyUI-SeedVR2_VideoUpscaler", {'install_reqs': True}),
]:
    image = image.run_commands([git_clone_cmd(repo, **flags)])

# FLUX Model download tasks
flux_model_tasks = [
    ("unet/FLUX", "flux1-dev-Q8_0.gguf", "city96/FLUX.1-dev-gguf", None),
    ("unet/FLUX", "flux-2-klein-9b-Q8_0.gguf", "unsloth/FLUX.2-klein-9B-GGUF", None),
    ("clip/FLUX", "t5-v1_1-xxl-encoder-Q8_0.gguf", "city96/t5-v1_1-xxl-encoder-gguf", None),
    ("clip/FLUX", "clip_l.safetensors", "comfyanonymous/flux_text_encoders", None),
    ("loras", "mjV6.safetensors", "strangerzonehf/Flux-Midjourney-Mix2-LoRA", None),
    ("vae/FLUX", "ae.safetensors", "ffxvs/vae-flux", None),
]

# Qwen-Image-Edit Model download tasks (CORRECTED PATHS)
qwen_model_tasks = [
    ("diffusion_models", "flux-2-klein-base-9b-fp8.safetensors", "black-forest-labs/FLUX.2-klein-base-9b-fp8", None),
    # Main Qwen-Image-Edit model - в підпапці split_files/diffusion_models
    ("diffusion_models", "Qwen-Image 2512_fp8_e5m2.safetensors", "art0123/Models_collection", "Qwen-Image-2512"),
    ("diffusion_models", "qwen_image_edit_2509_fp8_e4m3fn.safetensors", "Comfy-Org/Qwen-Image-Edit_ComfyUI", "split_files/diffusion_models"),
    ("diffusion_models", "qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning.safetensors", "lightx2v/Qwen-Image-Edit-2511-Lightning", None),
    ("diffusion_models", "qwen_image_edit_2511_fp8_e4m3fn.safetensors", "xms991/Qwen-Image-Edit-2511-fp8-e4m3fn", None),
    ("diffusion_models", "z_image_turbo_bf16.safetensors", "Comfy-Org/z_image_turbo", "split_files/diffusion_models"),
    ("text_encoders", "qwen_3_8b_fp4mixed.safetensors", "Comfy-Org/vae-text-encorder-for-flux-klein-9b", "split_files/text_encoders"),
    # Text encoder - в головній папці Qwen-Image_ComfyUI
    ("text_encoders", "qwen_2.5_vl_7b_fp8_scaled.safetensors", "Comfy-Org/Qwen-Image_ComfyUI", "split_files/text_encoders"),
    ("text_encoders", "Josiefied-Qwen3-8B-abliterated-v1.Q8_0.gguf", "mradermacher/Josiefied-Qwen3-8B-abliterated-v1-GGUF", ""),
    ("text_encoders", "qwen_3_4b.safetensors", "Comfy-Org/z_image_turbo", "split_files/text_encoders"),
    ("text_encoders", "qwen-4b-zimage-heretic-q8.gguf", "Lockout/qwen3-4b-heretic-zimage", None),
    {
        "subdir": "text_encoders",
        "filename": "qwen3_4b_thinking_2507.safetensors",
        "primary": {"url": "https://civitai.com/api/download/models/2563867?type=Model&format=SafeTensor&size=full&fp=bf16"}
    },

    # VAE model - в головній папці Qwen-Image_ComfyUI  
    ("vae", "flux2-vae.safetensors", "Comfy-Org/vae-text-encorder-for-flux-klein-9b", "split_files/vae"),
    ("vae", "qwen_image_vae.safetensors", "Comfy-Org/Qwen-Image_ComfyUI", "split_files/vae"),
    ("vae", "diffusion_pytorch_model.safetensors", "Owen777/UltraFlux-v1", "vae", "UltraFlux_vae.safetensors"),
    ("vae", "ae.safetensors", "Comfy-Org/z_image_turbo", "split_files/vae"),
    
    # Flux 2 Klein LoRAs
    ("loras/FLUX9bKlein", "The_Body_Version_A_Flux2.k.9B_r16_AdamW8Bit_Weighted_768_woman_000005000.safetensors", "andrewwe/klein9bl", None),
    ("loras/FLUX9bKlein", "The_Body_Version_M_Flux.2.klein.9B.r16._000005000.safetensors", "andrewwe/klein9bl", None),
    ("loras/FLUX9bKlein", "detail_slider_klein_9b_20260123_065513.safetensors", "andrewwe/klein9bl", None),
    ("loras/FLUX9bKlein", "f2_klein9b_macromastia_clothed.safetensors", "andrewwe/klein9bl", None),
    ("loras/FLUX9bKlein", "klein_slider_anatomy.safetensors", "andrewwe/klein9bl", None),
    ("loras/FLUX9bKlein", "nipplediffusion-f2-klein-9b.safetensors", "andrewwe/klein9bl", None),
    ("loras/FLUX9bKlein", "NSFW-klein.safetensors", "andrewwe/klein9bl", None),
    ("loras/FLUX9bKlein", "flux-object-remove-lora_comfy_converted.safetensors", "andrewwe/klein9bl", None),

    # Lightning LoRA models
    ("loras", "Qwen-Image-Lightning-4steps-V1.0.safetensors", "ModelTC/Qwen-Image-Lightning", None),
    ("loras", "Qwen-Image-Lightning-8steps-V1.0.safetensors", "ModelTC/Qwen-Image-Lightning", None),
    ("loras", "Qwen-Image-Lightning-4steps-V2.0.safetensors", "lightx2v/Qwen-Image-Lightning", None),
    ("loras", "Qwen-Image-Lightning-8steps-V2.0.safetensors", "lightx2v/Qwen-Image-Lightning", None),
    ("loras", "Wuli-Qwen-Image-2512-Turbo-LoRA-4steps-V2.0-bf16.safetensors", "Wuli-art/Qwen-Image-2512-Turbo-LoRA", None),
    ("loras", "Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors", "lightx2v/Qwen-Image-2512-Lightning", None),
    ("loras", "Qwen-Image-Edit-2509-Lightning-8steps-V1.0-fp32.safetensors", "andrewwe/qwLoras", None),
    ("loras", "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-fp32.safetensors", "lightx2v/Qwen-Image-Edit-2511-Lightning", None),

    # Додаткові Qwen LoRA моделі з HuggingFace
    ("loras", "Famegrid_Qwen_Lora_Standard_V1.5_RealSkinFix.safetensors", "PetruZetta/famegrid_qwen_lora", None),
    ("loras", "HMFemme_V1.safetensors", "bananas42/HMfemme", None),
    ("loras", "2168252_remove clothes3000.safetensors", "amethyst9/2168252", None),
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
    ("loras", "remove_her_cloths_qwen_edit_2509_v2_000005250.safetensors", "Sentinel7/qwen-image", "lora"),
    ("loras", "remove_clothing.safetensors", "TomaOmito/Qwen-Edit-2509-Lora-Remove-Clothing", None),
    ("loras", "Qwen-Image-Edit-Remove-Clothes_V.1.safetensors", "lingo/qwen-image-edit-fun-lora", None),
    ("loras", "eigen-banana-qwen-image-edit-2509-fp16-lora.safetensors", "eigen-ai-labs/eigen-banana-qwen-image-edit", None),
    ("loras", "NSFW Female Enhancer Qwen V0.3.safetensors", "andrewwe/qwLoras", None),
    ("loras/Zit", "BystyMega_b8nk.safetensors", "andrewwe/zitLoras", None),
    ("loras/Zit", "MariaBody_000001500.safetensors", "andrewwe/zitLoras", None),
    ("loras/Zit", "b3tternud3s_v2.safetensors", "andrewwe/zitLoras", None),
    ("loras/Zit", "big-nipples.safetensors", "andrewwe/zitLoras", None),
    ("loras/Zit", "bustywoman_bs9ex.safetensors", "andrewwe/zitLoras", None),
    ("loras/Zit", "christina_ch6tina.safetensors", "andrewwe/zitLoras", None),
    ("loras/Zit", "girls_zimage_g5r4l.safetensors", "andrewwe/zitLoras", None),
    ("loras/Zit", "BigNatsv2.safetensors", "andrewwe/zitLoras", None),
    ("loras/Zit", "EuropeanGirls.safetensors", "andrewwe/zitLoras", None),
    ("loras/Zit", "ZImage_CockShock.safetensors", "andrewwe/zitLoras", None),
    ("loras/Zit", "ZPenisHelper.safetensors", "andrewwe/zitLoras", None),
    ("loras/Zit", "psxAM_v1_ZITamateurLora.safetensors", "andrewwe/zitLoras", None),
    ("loras/Zit", "z_image_turbo_ukgirl.safetensors", "andrewwe/zitLoras", None),
    ("loras/Zit", "zimage_luisanudism2.safetensors", "andrewwe/zitLoras", None),
    ("loras/Zit", "reverse_cgirl_zitv3.safetensors", "andrewwe/zitLoras", None),
    ("loras/Zit", "hugepeniszimagev14000.safetensors", "andrewwe/zitLoras", None),
    ("loras/Zit", "Hendricks_ZIT_000001800.safetensors", "andrewwe/zitLoras", None),
    ("loras/Zit", "Hendricks_ZIT_000002400.safetensors", "andrewwe/zitLoras", None),
    ("loras/Zit", "Hendricks_ZIT_000002700.safetensors", "andrewwe/zitLoras", None),
    ("loras/Zit", "Hendricks_ZIT.safetensors", "andrewwe/zitLoras", None),
    ("loras/Zit", "girls_zimage_g5r4l_000001500.safetensors", "andrewwe/zitLoras", None),
    ("loras/Zit", "b3tternud3s_v3.safetensors", "andrewwe/zitLoras", None),
    ("loras/Zit", "amateur_photography_zimage_v1.safetensors", "andrewwe/zitLoras", None),
    ("loras/Zit", "The_Body_Version_A_ZIT.safetensors", "andrewwe/zitLoras", None),
    ("loras/Zit", "loradivaMarina5.safetensors", "andrewwe/zitLoras", None),
    ("loras/Zit", "loradivaMarina7.safetensors", "andrewwe/zitLoras", None),
    ("loras/Zit", "loradivaMarina9.safetensors", "andrewwe/zitLoras", None),
    ("loras/Zit", "myTeen10ep.safetensors", "andrewwe/zitLoras", None),
    ("loras/Zit", "HQphoto_7.safetensors", "andrewwe/zitLoras", None),
    ("loras/Zit", "HQphoto_6.safetensors", "andrewwe/zitLoras", None),
    ("loras/Zit", "HQphoto_8.safetensors", "andrewwe/zitLoras", None),
    ("loras/Zit", "hbm_v3hbm_bs4_2000.safetensors", "JustAnotherCibrarian/base_acne", "2185997/2476946", "Huge_Breasts_Mixv3.safetensors"),
    ("loras/Zit", "saggers_by_deedeemegadoodo_zimage_v1.safetensors", "UnifiedHorusRA/Theslicedbread2", "Sagging_Breasts_by_Deedeemegadoodo/ZImageTurbo"),
    ("loras/Zit", "FemaleFacePortraitsDetailedSkin-ZImage.safetensors", "UnifiedHorusRA/Theslicedbread", "Female_-_Face_Portraits_-_Detailed_Skin_-_Z-Image/ZImageTurbo"),
    ("loras/Zit", "RebelReal(z-image).safetensors", "UnifiedHorusRA/Theslicedbread", "RebelReal_Z-Image/ZImageTurbo"),
    ("loras/Zit", "SonyAlpha_ZImage.safetensors", "UnifiedHorusRA/Theslicedbread2", "Sony_Alpha_A7_III_Style/ZImageTurbo"),
    ("loras/Zit", "zit-m4crom4sti4-v5-deturbo-noadapt-21epoc-k3nk.safetensors", "K3NK/loras-zimageturbo", None),
    ("loras/Zit", "Reality Huge Breasts_p.safetensors", "UnifiedHorusRA/Theslicedbread", "Reality_Huge_Breasts_Z-image/ZImageTurbo"),
    ("loras/Zit", "Z-TURBO_Photography_35mmPhoto_1536.safetensors", "UnifiedHorusRA/Theslicedbread", "35mm_Photo_-_Flux_Z-Turbo/ZImageTurbo"),

    # LoRA-файли з wiikoo/Qwen-lora-nsfw
    ("loras", "reclining_nude_v1_000003500.safetensors", "wiikoo/Qwen-lora-nsfw", "loras"),
    ("loras", "consistence_edit_v2.safetensors", "wiikoo/Qwen-lora-nsfw", "loras2"),
    ("loras", "qwen_snofs.safetensors", "wiikoo/Qwen-lora-nsfw", "loras"),

    # Optional GGUF version for lower VRAM usage
    ("unet", "qwen-image-edit-2511-Q8_0.gguf", "unsloth/Qwen-Image-Edit-2511-GGUF", None),
    # SeedVR2 Upscaler Models
    ("SEEDVR2", "seedvr2_ema_3b-Q4_K_M.gguf", "cmeka/SeedVR2-GGUF", None),
    # Qwen-Rapid-AIO-NSFW-v17
    ("checkpoints", "Qwen-Rapid-AIO-NSFW-v17.safetensors", "Phr00t/Qwen-Image-Edit-Rapid-AIO", "v17"),
]


# Combine all model tasks
model_tasks = flux_model_tasks + qwen_model_tasks

extra_cmds = [
    f"wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth -P {MODELS_DIR}/upscale_models",
]

# Create volume
vol = modal.Volume.from_name("comfyui-app", create_if_missing=True)

app = modal.App(name="comfyui-l40s", image=image)

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

    # Fix detached HEAD and update ComfyUI backend to the latest version
    print("Fixing git branch and updating ComfyUI backend to the latest version...")
    os.chdir(DATA_BASE)
    try:
        update_git_repo(DATA_BASE, "ComfyUI backend")
    except Exception as e:
        print(f"Unexpected error during backend update: {e}")

    # Update ComfyUI-Manager to the latest version
    manager_dir = os.path.join(CUSTOM_NODES_DIR, "ComfyUI-Manager")
    if os.path.exists(manager_dir):
        print("Updating ComfyUI-Manager to the latest version...")
        os.chdir(manager_dir)
        try:
            update_git_repo(manager_dir, "ComfyUI-Manager")
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

    # Configure ComfyUI-Manager: Disable auto-fetch, set weak security, and disable file logging
    # CHANGED (V3.38+): Config path moved to user/__manager to avoid legacy migration issues
    # OLD: manager_config_dir = os.path.join(DATA_BASE, "user", "default", "ComfyUI-Manager")
    manager_config_dir = os.path.join(DATA_BASE, "user", "__manager")
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
        os.path.join(MODELS_DIR, "SEEDVR2"),           # For SeedVR2 models
        os.path.join(MODELS_DIR, "checkpoints"),       # For Checkpoints
        os.path.join(MODELS_DIR, "upscale_models"),
        TMP_DL
    ]
    
    for d in required_dirs:
        os.makedirs(d, exist_ok=True)

    # Download models at runtime (only if missing) - NOW INCLUDES QWEN
    print("Checking and downloading missing FLUX and Qwen-Image-Edit models...")
    for task in model_tasks:
        if isinstance(task, dict):
            # New smart download format
            download_model(
                subdir=task["subdir"],
                filename=task["filename"],
                primary_source=task["primary"],
                backup_source=task.get("backup"),
                local_filename=task.get("local_filename")
            )
        else:
            # Legacy tuple format
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
    print(f"Starting ComfyUI from {DATA_BASE} on {GPU_TYPE} with FLUX and Qwen-Image-Edit support...")
    
    # Start ComfyUI server with correct syntax and latest frontend
    cmd = ["comfy", "launch", "--", "--listen", "0.0.0.0", "--port", "8000"]
    print(f"Executing: {' '.join(cmd)}")
    
    subprocess.Popen(
        cmd,
        cwd=DATA_BASE,
        env=os.environ.copy()
    )
