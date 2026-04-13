Audience: engineer
Use context: staging
Priority: correctness
Definition of done: new v3 Modal script exists with visible startup issues fixed, v2 remains untouched, notebook points to v3, changes are verified and prepared for GitHub push
Avoid: hacks, filler, overengineering, fake completeness
Mode: calm, direct, specific

- [x] Inspect current `v2` script, startup log, and git state.
- [x] Create new `v3` script with the correct GPU in the filename.
- [x] Fix visible startup issues in `v3` without changing `v2`.
- [x] Update the Colab notebook to deploy the new `v3` script.
- [x] Verify syntax and review the final diff.
- [ ] Commit the targeted changes and push them to GitHub.

Notes
- `comfyui_app_a100_v2.py` must remain unchanged during this iteration.
- The repo already has unrelated local changes; only stage the targeted files.

Summary
- Added `comfyui_app_l40s_v3.py` as the new editable launcher and left `v2` untouched.
- Hardened runtime git update logic, removed repeated runtime pip/comfy-cli upgrades, cached frontend requirements sync by hash, increased startup/runtime timeouts, and ensured `upscale_models` exists before downloads.
- Switched the Colab notebook to download and deploy `comfyui_app_l40s_v3.py`.
