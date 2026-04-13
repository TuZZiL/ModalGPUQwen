Audience: engineer
Use context: staging
Priority: correctness
Definition of done: new v4 Modal script exists with the QuickPod `flux2_klein9b` model/node inventory, v3 remains untouched, notebook points to v4, changes are verified and pushed
Avoid: hacks, filler, overengineering, fake completeness
Mode: calm, direct, specific

- [x] Extract the exact model and custom-node inventory from `quick_download_quickpod_codex_v2.sh`.
- [x] Create new `v4` Modal script named with GPU and base model: `l40s_flux2_klein9b`.
- [x] Sync the `v4` script to the QuickPod inventory without editing `v3`.
- [x] Update the Colab notebook to deploy the new `v4` script.
- [x] Verify syntax and review the final diff.
- [ ] Commit the targeted changes and push them to GitHub.

Notes
- `comfyui_app_l40s_v3.py` must remain unchanged during this iteration.
- The repo already has unrelated local changes; only stage the targeted files for `v4`.

Summary
- Added `comfyui_app_l40s_flux2_klein9b_v4.py` as the new Modal launcher for `L40S + flux2_klein9b`.
- Synced `v4` exactly to the QuickPod reference inventory: 4 custom node repos and the full `flux2_klein9b` model list from `quick_download_quickpod_codex_v2.sh`.
- Removed the `v3`-specific broader Qwen/Z-image inventory and manager-specific runtime behavior from the new `v4` launcher.
- Switched the Colab notebook to download and deploy `comfyui_app_l40s_flux2_klein9b_v4.py`.
