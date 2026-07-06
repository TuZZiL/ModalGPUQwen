# 2026-07-06
- done: додано дві нові LoRA (`gab1car_Gabbie_v1_c1-st2000.safetensors` та `gab1car_Gabbie_v1_c1-st3000.safetensors`) до `model_tasks` у `comfyui_app_l40s_krea2_turbo_v2.py`.
- done: перевірено синтаксис скрипта.
- done: pushed to git.

# 2026-07-05
- done: додано дві нові LoRA (`Penis_KreaTurbo_v1-st3000.safetensors` та `Penis_KreaTurbo_v1-st5000.safetensors`) до `model_tasks` у `comfyui_app_l40s_krea2_turbo_v2.py`.
- done: перевірено синтаксис скрипта.
- done: pushed to git.


# 2026-07-04
- done: додано новий VAE (`vae_wan_2.1_vae.safetensors`) та LoRA (`krea2_realism_lora.safetensors`) до `model_tasks` у `comfyui_app_l40s_krea2_turbo_v2.py`.
- done: додано нову LoRA (`Better_Pussy_Poses_v3.0_Krea2.safetensors` з очищеним іменем) до `model_tasks` у `comfyui_app_l40s_krea2_turbo_v2.py`.
- done: додано diffusion model (`pornmasterKrea2_v1FP8.safetensors`) та LoRA (`Busty_KreaTurbo_v1-st3000.safetensors`) до `model_tasks` у `comfyui_app_l40s_krea2_turbo_v2.py`.
- done: перевірено синтаксис скрипта.
- done: pushed to git.



# 2026-07-03
- done: додано 3 нові LoRAs (`HMCum_krea2_epoch30.safetensors`, `RealisticSnapshotKrea2.safetensors`, `impreal.safetensors`) для Krea 2 до `model_tasks` у `comfyui_app_l40s_krea2_turbo_v2.py`.
- done: pushed to git.


# 2026-07-02
- done: оновлено `CUSTOM_NODE_REPOS` — замінено ноду на `TuZZiL/ComfyUI-ConditioningKrea2Rebalance` для Krea 2.
- done: додано 3 нові LoRAs до `model_tasks` (`fedor_bypass.safetensors`, `refiner_neuter_patch.safetensors`, `Krea2-realism-V2.safetensors`).
- done: оптимізовано завантаження моделей у `comfyui_app_l40s_krea2_turbo_v2.py` — всі посилання переведено з повільного `wget` на швидкий багатопотоковий `hf_hub_download` (hf_transfer).
- done: створено `comfyui_app_l40s_krea2_turbo_v2.py` з оптимізаціями cold start (~35s економії):
  - видалено runtime pip/comfy-cli upgrade → baked в image (~9s)
  - замінено `update_comfyui_frontend` на `sync_frontend_requirements` з SHA256-кешем (~23s)
  - зафіксовано frontend version `1.47.6` замість `@latest` (~1-2s)
  - паралельні git pulls для custom nodes через ThreadPoolExecutor (~3-4s)
- done: volume name `comfyui-krea2` не змінено — моделі не перекачуються.
- done: pushed to git.
- next: запуск v2 на Modal, порівняння часу cold start з v1.

# 2026-07-01
- done: додано 4 нові LoRAs (`skc3vo.safetensors`, `z0jglf.safetensors`, `snofs_krea_v1.safetensors`, `KNPV4.1_pre.safetensors`) для Krea 2 до `model_tasks` у `comfyui_app_l40s_krea2_turbo.py`.
- done: додано кастомну ноду `capitan01R/ComfyUI-Krea2T-Enhancer` до `CUSTOM_NODE_REPOS` для покращення дотримання промпту.
- done: додано новий VAE `krea2RealVae_v10.safetensors` до `model_tasks` у `comfyui_app_l40s_krea2_turbo.py`.
- done: видалено модель `krea2_turbo_int8_convrot.safetensors` з `model_tasks` за запитом.
- done: успішно перевірено синтаксис скрипта.
- next: запуск comfyui_app_l40s_krea2_turbo.py на Modal для перевірки завантаження нових моделей та ноди.

# 2026-06-30
- done: додано ще 3 додаткові LoRAs (разом 16 нових LoRAs) для Krea 2 до `model_tasks` у `comfyui_app_l40s_krea2_turbo.py`.
- done: оновлено версію LoRA `MysticXXX_KREA2_v1.safetensors` на `MysticXXX_KREA2_v2.safetensors` у `model_tasks`.
- done: додано репозиторій `erosDiffusion/ComfyUI-EulerDiscreteScheduler` до `CUSTOM_NODE_REPOS` для усунення помилки бракуючої ноди `ComfyUI-EulerFlowMatchingDiscreteScheduler`.
- done: перевірено синтаксис файлу (`py_compile`).
- next: запуск comfyui_app_l40s_krea2_turbo.py на Modal для перевірки завантаження LoRA та ноди.

# 2026-06-28
- done: додано кастомну ноду `ClownsharkBatwing/RES4LYF` до `CUSTOM_NODE_REPOS` у `comfyui_app_l40s_flux2_klein9b_v4.py`.
- next: перевірка завантаження ноди на Modal.

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
