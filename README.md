# Multimodal Deepfake Detection

An explainable, multimodal deepfake detection system that classifies social-media videos into six categories — **real**, **AI-generated video**, **faceswap**, **edited** (Face2Face/NeuralTextures), **voiceclone** and **audio-video fake** — and produces a per-video forensic report (decision, category, confidence, risk level and supporting evidence) through a Streamlit dashboard, a batch runner and a command-line tool.

📄 **Full technical details — architecture, dataset stats, training configuration, evaluation results and known limitations — are in [PROJECT_DOCUMENTATION.md](./PROJECT_DOCUMENTATION.md).**

---

## Dashboard

```powershell
streamlit run app\streamlit_app.py
```

| Page | What it does |
|---|---|
| **Analyze** | Upload one video → verdict banner, KPI cards, Plotly probability gauges, per-frame chart, face-crop gallery, audio window chart, downloadable forensic report |
| **Batch Analysis** | Score up to 40 files (upload or server folder) → verdict breakdown donut, per-file fake-score chart, sortable detail table, **CSV + report export** |
| **Model Metrics** | 6-class confusion-matrix heatmap, per-class precision/recall/F1 bars, support chart, headline metric cards and a side-by-side comparison of every evaluation run |
| **About** | Pipeline diagram, model inventory, dataset table, repository layout and the known generalisation limitation |

Styling lives in `.streamlit/config.toml` (dark theme) plus `app/ui/theme.py` (custom CSS: hero, KPI cards, verdict banners).

---

## Features

- YOLOv8-based face detection and cropping (frames converted BGR→RGB to match training)
- Two model families:
  - **6-class** ResNet18 (`detect_social_video.py`), with class-imbalance handling via `WeightedRandomSampler`
  - **Binary** video (ResNet18) + audio (Wav2Vec2) pair used by the dashboard — each modality reports an independent REAL/FAKE probability, combined into a CONSISTENT / INCONCLUSIVE status
- Audio scored over several overlapping 4-second windows and averaged
- Explainable forensic reports with per-frame, per-window and face-crop evidence
- Analysis engine (`app/core/`) is **Streamlit-free**, so the same code powers the dashboard, batch mode and scripts

## Project Status

- ✅ Core visual detection pipeline: trained, evaluated (95% validation accuracy), and deployed
- ✅ CLI, dashboard, batch runner and metrics page: fully connected to the trained models
- ⏸️ Known limitation: reduced accuracy on real-world/in-the-wild mobile video (see [Section 6, PROJECT_DOCUMENTATION.md](./PROJECT_DOCUMENTATION.md#6-known-limitation-cross-dataset-generalization-gap)) — DFDC dataset integration planned to address this
- ✅ Audio branch (Wav2Vec2): retrained on **video-domain** audio and wired into the dashboard — 87.8% on held-out videos (the original ASVspoof-trained head scored 26.8% there). Original checkpoint kept as `saved_models/audio_binary_small_asvspoof.pth`
- ⏸️ Fusion model: trained but the dashboard intentionally shows video and audio independently (no weighted 60/40 blend)

---

## Setup

```powershell
git clone <repo-url>
cd multimodal-deepfake-detection

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

**Note:** Dataset files, trained model weights and YOLO weights are excluded from git (size/licensing, see `.gitignore`). Contact a team member for `saved_models/social_media_detector.pth`, `saved_models/video_binary_small.pth`, `saved_models/audio_binary_small.pth` and `weights/yolov8n-face.pt`, or rebuild them from source following [PROJECT_DOCUMENTATION.md](./PROJECT_DOCUMENTATION.md#3-dataset).

> **Run everything with the same interpreter** that has your dependencies. If `streamlit` on your PATH belongs to a different Python you get import errors such as `cannot import name 'Wav2Vec2Model' from 'transformers'`. Check with:
> ```powershell
> python -c "import sys, transformers; print(sys.executable); from transformers import Wav2Vec2Model; print('OK')"
> ```

Audio extraction needs **ffmpeg**. It is discovered automatically from (1) the `FFMPEG_BINARY` environment variable, (2) `PATH`, (3) known install directories — see `app/core/config.py`.

---

## Usage

**Launch the dashboard**
```powershell
streamlit run app\streamlit_app.py
```

**Run inference on a video (CLI, 6-class)**
```powershell
python detect_social_video.py <video_file.mp4>
```

**Train**
```powershell
python -m training.train_video            # 6-class video model
python -m training.train_audio_v2         # audio head
python -m training.train_fusion           # fusion head (optional)
```

**Rebuild the audio dataset / retrain the video-domain audio head**
```powershell
python -m scripts.build_video_audio_dataset   # label + extract from dataset/ videos
python -m scripts.train_audio_video_domain    # retrains head, backs up old checkpoint
```

**Verify the whole pipeline end to end**
```powershell
python -m scripts.verify_app                  # status combos, video, audio, report
python -m scripts.evaluate_audio_on_videos    # audio accuracy on real video files
```

---

## Project Structure

```
multimodal_deepfake_detection/
├── .streamlit/
│   └── config.toml          # dark theme, upload limits, server config
│
├── detect_social_video.py   # CLI entry point (6-class)
│
├── app/                     # ── Streamlit dashboard ──
│   ├── streamlit_app.py     # Analyze page (entry point, ~150 lines)
│   ├── pages/
│   │   ├── 1_Batch_Analysis.py
│   │   ├── 2_Model_Metrics.py
│   │   └── 3_About.py
│   ├── core/                # analysis engine — no Streamlit imports
│   │   ├── config.py        #   paths, constants, ffmpeg discovery
│   │   ├── models.py        #   architectures + checkpoint loading
│   │   ├── video.py         #   frame sampling, face crop, scoring
│   │   ├── audio.py         #   extraction, windowing, scoring
│   │   ├── verdict.py       #   status rules + report builder
│   │   ├── analyze.py       #   end-to-end orchestrator
│   │   └── metrics.py       #   evaluation-report parsers
│   └── ui/                  # presentation layer
│       ├── theme.py         #   custom CSS, KPI cards, verdict banner
│       ├── charts.py        #   Plotly figure builders
│       ├── components.py    #   result rendering (tabs, evidence)
│       ├── sidebar.py       #   shared sidebar
│       └── state.py         #   cached model loader
│
├── models/                  # model architecture definitions
├── preprocessing/           # frame extraction, face cropping, audio extraction
├── training/                # training + evaluation entry points
├── scripts/                 # dataset build, checkpoint training, verification
├── utils/                   # dataset summary helper
│
├── results/                 # evaluation reports (committed)
│   ├── video_evaluation.txt
│   └── 6class_video_confusion_matrix.txt
│
├── saved_models/            # checkpoints                (gitignored)
├── weights/                 # YOLO face detector         (gitignored)
├── dataset/                 # raw datasets               (gitignored)
├── processed/               # extracted frames/audio     (gitignored)
│
├── requirements.txt
├── README.md
└── PROJECT_DOCUMENTATION.md # full technical report
```

---

## Results

| Evaluation | Accuracy | Macro F1 |
|---|---|---|
| 6-class video model (held-out validation, 3,717 images) | **95.0%** | 0.89 |
| Video-only on fusion labels (34 videos) | 79.4% | 0.63 |
| Video + audio fusion (34 videos) | 73.5% | 0.69 |
| Audio branch on held-out video audio (41 clips) | **87.8%** | — |

Per-class validation scores and the full confusion matrix are rendered live on the **Model Metrics** page.

---

## Known Limitation

The 6-class model was validated at 95% accuracy **on data drawn from the same distribution it was trained on** (FaceForensics++, Celeb-DF, FakeAVCeleb — landscape, medium-shot, c40-compressed footage). On in-the-wild vertical mobile video it can label manipulated content as `real` with high confidence. This is a distribution-shift problem, not a code defect. See [Section 6, PROJECT_DOCUMENTATION.md](./PROJECT_DOCUMENTATION.md#6-known-limitation-cross-dataset-generalization-gap).

A `REAL` verdict means **"not detected"** — never proof of authenticity.

---

## Team

Maintained by the Multimodal Deepfake Detection team.
