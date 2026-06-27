# 2026-06-27
- done: додано новий uncensored текстовий енкодер `Qwen3-8B-Gemini-2.5-Flash-Uncensored-Q8_0.gguf` до `model_tasks` у `comfyui_app_l40s_flux2_klein9b_v4.py`.
- done: додано 6 нових лор (SexGod, SmartCharacterSwap, NaturalBeauty, 34O CUP MACROMASTIA, C.H.E.S.T., Pussy FIX) до `model_tasks`.
- done: успішно перевірено синтаксис скрипта.
- next: перевірка завантаження нових моделей на Modal.

# 2026-06-26
- done: додано 6 нових лор (HugeDick_c1-st3000/4000, TribeW_v1_c1-st3000/4000/5000/6000) до `model_tasks` у `comfyui_app_l40s_flux2_klein9b_v4.py`.
- done: успішно перевірено синтаксис скрипта.
- next: перевірка завантаження лор на Modal.

# 2026-06-23
- done: створено окремий Modal-скрипт `comfyui_app_l40s_krea2.py` для запуску Krea 2 Base/Turbo.
- done: додано завантаження моделей `Krea2_Turbo_fp8mixed.safetensors`, `qwen3vl_4b_fp8_scaled.safetensors` та VAE `qwen_image_vae.safetensors`.
- done: підключено кастомні ноди `ComfyUI-ConditioningKrea2Rebalance`, `Winnougan/WINT8-ComfyUI` та `ltdrdata/ComfyUI-Manager` (для гарантованого встановлення менеджера через git clone).
- done: додано відсутні pip-залежності (`scikit-image`, `ultralytics`, `webcolors`, `beautifulsoup4`) до Modal образу для запобігання збоїв завантаження нод.
- next: перевірка працездатності скрипта на Modal.

# 2026-04-26
- done: додано лору `Flat_C_6.safetensors` до `model_tasks` у `comfyui_app_l40s_flux2_klein9b_v4.py`.
- done: додано 4 нові лори (`Remove_DressMax_9.safetensors`, `Remove_DressMax_8.safetensors`, `Remove_DressMax_7.safetensors`, `Flat_C_2.safetensors`) до `model_tasks` у `comfyui_app_l40s_flux2_klein9b_v4.py`.
- next: перевірка в ComfyUI.

# 2026-04-21
- done: додано лору `SheerSeeThrough_F2K9B_v1.safetensors` до `model_tasks`.
- done: додано 2 лори (`Chest_9Bslider.safetensors`, `buttocksslider.safetensors`) до `model_tasks` у `comfyui_app_l40s_flux2_klein9b_v4.py`
- next: перевірка в ComfyUI.

# 2026-04-20
- done: додано 3 нові лори до `model_tasks` у `comfyui_app_l40s_flux2_klein9b_v4.py`:
  - `Flux2-Klein-9B-consistency-V2.safetensors`
  - `klein_snofs_v1_3.safetensors`
  - `remove_dress1904_4.safetensors`
  - `pussydiffusion-f2-klein-9b_v2.safetensors`
  - `Realism_Engine_Klein_V1.safetensors`
  - `klein-m4crom4sti4-v2-3epoc-k3nk.safetensors`
- done: оновлено `eros_fklein` до версії `v2_000019795` (версію `v1` видалено).
- done: видалено 4 застарілі лори (замінено новими версіями або видалено за запитом):
  - `klein_snofs_v1_1.safetensors`
  - `ChrisHendricks_v2_4.safetensors`
  - `SEXGOD_ImprovedNudity_Klein9b_v2_5.safetensors`
  - `pussydiffusion-f2-klein-9b.safetensors`
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
