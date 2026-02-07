# Frozen VLM Baselines + Dual-Encoder Fusion for CheXpert

This repository scaffolds a reproducible, Google Colab Pro–friendly workflow for chest X-ray VLM evaluation on CheXpert (5-label) with external zero-shot generalization. The MVP includes:

1. Kaggle download + manifest + QC for CheXpert.
2. EDA-lite plots saved to Drive.
3. Zero-shot CLIP baseline on CheXpert with metrics + curves.

> **Important**: Encoder weights are always frozen. Training stages beyond the MVP are scaffolded and will be implemented after MVP passes.

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

Artifacts will be written to `runs/<run_id>` and mirrored to `/content/drive/MyDrive/FrozenVLM-CheXpert/runs/<run_id>`.

## Project Structure

```
/project
  /configs
  /data
  /src
    /datasets
    /models
    /training
    /eval
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
