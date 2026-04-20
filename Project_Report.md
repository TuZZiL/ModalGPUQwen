# 2026-04-20
- done: додано 3 нові лори до `model_tasks` у `comfyui_app_l40s_flux2_klein9b_v4.py`:
  - `Flux2-Klein-9B-consistency-V2.safetensors`
  - `klein_snofs_v1_3.safetensors`
  - `remove_dress1904_4.safetensors`
- done: перевірено синтаксис `python -m py_compile`.
- next: моніторинг стабільності завантаження при старті App.

# 2026-04-19
- done: додано лору `remove_dress1904_3.safetensors` до `model_tasks` у `comfyui_app_l40s_flux2_klein9b_v4.py`.
- done: перевірено синтаксис `python -m py_compile`.
- next: працювати лише в `comfyui_app_l40s_flux2_klein9b_v4.py`, вносити surgical changes.
- context: файл керує Modal image, runtime update ComfyUI, sync custom nodes та завантаженням моделей.
- summary: додано нову лору за запитом користувача.

- done: замінено custom node repo з `TuZZiL/Comfyui-flux2klein-Lora-loader` на `TuZZiL/tuz-fluxklein-toolkit`.
- done: перевірено, що зміна локальна до `CUSTOM_NODE_REPOS`.
- done: `python -m py_compile comfyui_app_l40s_flux2_klein9b_v4.py` проходить.
- next: commit/push тільки релевантної зміни.
