# Lessons

## 2026-04-13
- When a Modal launcher script is versioned forward, keep the previous version file immutable and land fixes in the new versioned file only.
- Runtime `git pull` logic inside a persistent ComfyUI volume must not hardcode `origin/main`; detect the remote default branch or skip safely.
- Expensive bootstrap steps that do not need to run on every cold start should be baked into the image or gated by a change detector.
- When one environment script becomes the reference inventory for models and custom nodes, derive the other launcher from that inventory exactly instead of maintaining two drifting supersets.
- If a runtime probe fails on a standard dependency like `blake3`, add it to the build image and keep the probe as a guard rather than trying to paper over the import crash.
