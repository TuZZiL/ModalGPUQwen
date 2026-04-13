# Lessons

## 2026-04-13
- When a Modal launcher script is versioned forward, keep the previous version file immutable and land fixes in the new versioned file only.
- Runtime `git pull` logic inside a persistent ComfyUI volume must not hardcode `origin/main`; detect the remote default branch or skip safely.
- Expensive bootstrap steps that do not need to run on every cold start should be baked into the image or gated by a change detector.
- When one environment script becomes the reference inventory for models and custom nodes, derive the other launcher from that inventory exactly instead of maintaining two drifting supersets.
- If a runtime probe fails on a standard dependency like `blake3`, add it to the build image and keep the probe as a guard rather than trying to paper over the import crash.
- For dependency diagnosis, prefer `importlib.metadata.version(...)` over import probes when the import itself may be crashing the app before the actual issue is visible.
- When `torchvision::nms` crashes at import time, treat the `torch / torchvision / torchaudio` trio as a pinned unit and reinstall them together from the matching wheel index.
- A single missing custom-node dependency like `piexif` can be fixed directly in the build image even if the node itself is optional and ComfyUI still starts.
- When borrowing upstream update logic, restore the whole update path coherently: backend, manager, frontend, and launch flags should move together or the flow becomes harder to reason about.
- A `403 Forbidden` on `GET /` behind a reverse proxy is often a host/origin protection issue; `--enable-cors-header` is a targeted ComfyUI-side mitigation worth trying before changing proxy layers.
