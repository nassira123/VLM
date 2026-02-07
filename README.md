# Frozen VLM Baselines + Dual-Encoder Fusion for CheXpert

This repository provides a Colab Pro–friendly MVP workflow for chest X-ray VLM evaluation on CheXpert (5-label). The MVP is **fully runnable** end-to-end in one Colab session:

1. Kaggle download + CheXpert manifest + QC.
2. EDA-lite plots (label counts, uncertain counts, image size histogram, sample grid).
3. Zero-shot CLIP baseline on CheXpert with metrics and ROC/PR plots.

> **Important**: Encoder weights are always frozen. Training stages beyond the MVP are intentionally not implemented yet.

## Colab Quickstart (CLI)

1. Mount Drive in Colab:

```python
from google.colab import drive
drive.mount('/content/drive')
```

2. Configure Kaggle credentials (upload `kaggle.json` and run):

```bash
mkdir -p ~/.kaggle
cp /content/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the MVP pipeline:

```bash
python scripts/prepare_data.py --config configs/data.yaml
python scripts/run_eda.py --config configs/eda.yaml
python scripts/eval_zeroshot.py --config configs/eval_zeroshot.yaml
```

Artifacts are written to `runs/<run_id>` and a convenience symlink `runs/latest` is updated after each run. When Drive is mounted, artifacts are mirrored to:

```
/content/drive/MyDrive/FrozenVLM-CheXpert/runs/<run_id>/
```

## Expected Outputs (MVP)

- `runs/<run_id>/tables/manifest_chexpert.csv`
- `runs/<run_id>/tables/qc_report.json`
- `runs/<run_id>/plots/*.png` (EDA + ROC/PR curves)
- `runs/<run_id>/tables/metrics.json`, `auc.csv`, `threshold_metrics.csv`
- `runs/<run_id>/run_summary.json`
- `runs/<run_id>/configs/resolved_config.yaml`

## Project Structure

```
/project
  /configs
  /data
  /src
    /datasets
    /eda
    /eval
    /models
    /training
    /utils
  /scripts
  /reports
  requirements.txt
  README.md
```

## Notes
- CheXpert uses the official patient-wise train/valid split; manifest includes patient_id, study_id, view, and raw labels.
- Thresholds are fixed at 0.2, 0.5, 0.8 for all metrics (no tuning).
- External datasets (Pneumonia, COVID) are downloaded but not yet integrated into evaluation scripts.

## Next Steps After MVP
- Add BiomedCLIP zero-shot baseline.
- Implement linear probe on frozen features.
- Implement Project B dual-encoder fusion with dynamic soft prompts.
- Add ablations and report generation.
