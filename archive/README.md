# Archive

This folder contains deprecated or legacy code and documentation that is no longer part of the main release. Files here are kept for reference and reproducibility.

**Contents:**
- `old_main.py` — earlier simulation version (current training uses `train_env.py` in root)
- `a.py`, `text.py`, `naive_faster.py` — utility scripts
- `validate_fix.py` — one-off validation script (imports `train_env` from parent)
- `sharp_3d_surface.py`, `smooth_3d_surface.py`, `vis.py` — visualization experiments
- Various fix/checklist markdown files from development

**Note:** To run `validate_fix.py`, execute from the repo root: `python archive/validate_fix.py`
