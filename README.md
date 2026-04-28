# ModalGPUQwen

Запуск [ComfyUI](https://github.com/comfyanonymous/ComfyUI) та AI-інструментів на [Modal.com](https://modal.com/) — хмарних GPU без необхідності налаштовувати сервери вручну.

## Можливості

- **FLUX 2 Klein 9B** — основний стек генерації зображень (GGUF Q8_0 + 50+ LoRA адаптерів)
- **Cosmos Predict2 14B** — text-to-image через ComfyUI workflow
- **AI Toolkit** — тренування / fine-tuning LoRA моделей (Gradio UI)
- **Різні GPU** — A10G, L40S, A100, H100 — окремі конфігурації під кожний

## Структура проекту

| Файл | Призначення | GPU |
|---|---|---|
| `comfyui_app_l40s_flux2_klein9b_v4.py` | **ComfyUI + FLUX 2 Klein 9B** (основний стек, ~60 LoRA) | L40S |
| `comfyui_app_a100_v2.py` | ComfyUI (A100 версія) | A100 |
| `comfyui_app_a100.py` | ComfyUI (A100, розширений набір нод та моделей) | L40S |
| `comfyui_app_a10g.py` | ComfyUI (полегшена версія) | A10G |
| `comfyui_app_h100.py` | ComfyUI (H100 версія) | H100 |
| `comfyui_app_l40s_v3.py` | ComfyUI (L40S, рання версія) | L40S |
| `ai_toolkit_app_a100.py` | AI Toolkit — тренування LoRA (Gradio) | A100 |
| `clone_node.py` | Клонування кастомних нод у Modal Volume | — |
| `comfyui_modal.ipynb` | Colab ноутбук для деплою ComfyUI | — |
| `ai_toolkit_modal.ipynb` | Colab ноутбук для деплою AI Toolkit | — |
| `workflow/` | ComfyUI workflow файли | — |
| `quickpod/` | Скрипти для QuickPod GPU instances | — |
| `tasks/` | Нотатки, TODO, troubleshooting | — |

## Швидкий старт

### 1. Локальний запуск

```bash
# Встановити Modal CLI
pip install modal

# Авторизуватися
modal token set

# Деплоїти ComfyUI з FLUX 2 Klein 9B
modal deploy comfyui_app_l40s_flux2_klein9b_v4.py
```

### 2. Через Google Colab

Відкрийте `comfyui_modal.ipynb` в Google Colab — він встановить залежності та задеплоїть додаток автоматично.

### 3. QuickPod Scripts

Для завантаження моделей на QuickPod GPU instances:

```bash
cd /workspace
python download_flux2_klein_quickpod.py
```

Детальніше: [quickpod/README.md](./quickpod/README.md)

## Посилання

- [Modal.com](https://modal.com/) — платформа для запуску GPU
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) — графічний редактор для генерації зображень
- [FLUX 2 Klein 9B](https://huggingface.co/unsloth/FLUX.2-klein-9B-GGUF) — модель на Hugging Face
- [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager) — менеджер кастомних нод

## Туторіали

- [ComfyUI на Modal.com](https://www.marwanto606.com/2025/05/cara-menjalankan-comfyui-di-modal-dot-com.html)
- [Cosmos Predict2 на Modal](https://www.marwanto606.com/2025/06/cara-menjalankan-nvidia-cosmos-predict2-di-comfyui-modal-dot-com.html)