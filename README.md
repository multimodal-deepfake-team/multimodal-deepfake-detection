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
- Leak-free evaluation protocol: video- and group-level splits, a leakage check script, video-level metrics and source-bias diagnostics (see [Evaluation Protocol](#evaluation-protocol-leak-free))

## Project Status

- ✅ Core visual detection pipeline: trained and deployed. The original 95% validation accuracy was measured on a leaky image-level split; on a leak-free video-level split the 5-class model reaches **91.0%** (see [Results](#results))
- ✅ CLI, dashboard, batch runner and metrics page: connected to the originally trained models (the new leak-free 5-class model `video_model_v3_best.pth` is not yet wired into the dashboard)
- ✅ Leak-free split (`splits/v2.csv`), leakage check, source-wise accuracy, compression sweep and miss diagnostics
- ⏸️ Known limitation: reduced accuracy on real-world/in-the-wild mobile video (see [Section 6, PROJECT_DOCUMENTATION.md](./PROJECT_DOCUMENTATION.md#6-known-limitation-cross-dataset-generalization-gap)). A cross-dataset evaluation on an in-the-wild benchmark is planned
- ✅ Audio branch (Wav2Vec2): retrained on **video-domain** audio and wired into the dashboard — 87.8% on held-out videos (the original ASVspoof-trained head scored 26.8% there). Original checkpoint kept as `saved_models/audio_binary_small_asvspoof.pth`
- ⏸️ Fusion model: trained but the dashboard intentionally shows video and audio independently (no weighted 60/40 blend). The fusion evaluation set (34 videos) is too small for strong claims

---

## Setup

```powershell
git clone <repo-url>
cd multimodal-deepfake-detection

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

**Note:** the three checkpoints the dashboard needs **ship with the repo** — `saved_models/video_binary_small.pth`, `saved_models/audio_binary_small.pth` and `weights/yolov8n-face.pt` are committed on purpose (see `.gitignore`), so a fresh clone can run the app straight away. Everything else — datasets, the 6-class CLI model (`saved_models/social_media_detector.pth`), the leak-free v3 model, the fusion head — is still excluded from git (size/licensing). Contact a team member or rebuild those from source following [PROJECT_DOCUMENTATION.md](./PROJECT_DOCUMENTATION.md#3-dataset).

> `requirements.txt` installs the **CPU** build of PyTorch on every platform so the same file works on Streamlit Community Cloud. If you have a GPU and want CUDA for training, swap it in afterwards with:
> ```powershell
> uv pip install --force-reinstall torch==2.14.0 torchvision==0.29.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu126
> ```

> **Run everything with the same interpreter** that has your dependencies. If `streamlit` on your PATH belongs to a different Python you get import errors such as `cannot import name 'Wav2Vec2Model' from 'transformers'`. Check with:
> ```powershell
> python -c "import sys, transformers; print(sys.executable); from transformers import Wav2Vec2Model; print('OK')"
> ```

Audio extraction needs **ffmpeg**. It is discovered automatically from (1) the `FFMPEG_BINARY` environment variable, (2) `PATH`, (3) known install directories — see `app/core/config.py`.

---

## Deploy to Streamlit Community Cloud

The repository is deployment-ready: push `main` and point Community Cloud at the entry point.

1. Push the repository to GitHub (the three checkpoints are committed, so no extra upload step).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **Create app** → *“Yup, I have an app”* and fill in:
   - **Repository:** `multimodal-deepfake-team/multimodal-deepfake-detection`
   - **Branch:** `main`
   - **Main file path:** `app/streamlit_app.py`
3. Open **Advanced settings** and pick a **Python version** (`3.12`, `3.13` and `3.14` are all verified to resolve — the default 3.12 is fine). No secrets are required.
4. **Deploy.** The first build installs the pinned dependencies; after that every push to `main` redeploys automatically.

### What the deployment files do

| File | Why it exists |
|---|---|
| `requirements.txt` | Pins every dependency to the versions the dashboard was verified on and adds `--extra-index-url https://download.pytorch.org/whl/cpu`, so the cloud build resolves `torch==2.14.0+cpu` (a CPU-only wheel) instead of a CUDA-enabled one. Resolution was checked with `uv` (Community Cloud's resolver) for Linux on Python 3.12/3.13/3.14. |
| `packages.txt` | apt packages for the non-Python dependencies: `ffmpeg` (audio branch) plus `libgl1` / `libglib2.0-0` / `libsm6` / `libxext6` / `libxrender1`, because ultralytics pulls the GUI build of `opencv-python` in next to the headless one and its `cv2` module needs libGL at import time. |
| `saved_models/*.pth`, `weights/yolov8n-face.pt` | The runtime checkpoints (≈51 MB total, well under GitHub's 100 MB per-file limit). `.gitignore` carries explicit `!` carve-outs for exactly these three files; every other checkpoint stays ignored. |
| `processed/fusion_evaluation.txt` | A text report the **Model Metrics** page reads. It is committed like the artefacts in `results/`. |

### Runtime behaviour in the cloud

- The runtime is CPU-only with roughly 1–3 GB of RAM, so a 10-second clip takes on the order of 20–60 s to score — the app shows a progress spinner, not a hang.
- The **first analysis** downloads `facebook/wav2vec2-base` (≈360 MB) from the Hugging Face Hub into the container's cache (~1 min). It is not written to the repository and is reused by later analyses and sessions.
- Uploaded videos are processed in a temporary directory and deleted immediately after scoring; nothing is persisted.
- If a checkpoint is ever missing, the Analyze and Batch pages name the exact file instead of failing deep inside the pipeline, and **Model Metrics** / **About** keep working.

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
python -m training.train_video            # original 6-class video model (leaky image-level split)
python training\train_video_v3.py         # 5-class video model on the leak-free split (splits/v2.csv)
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

## Evaluation Protocol (leak-free)

The original validation split was a random 80/20 split of face crops, so crops from the same video could land in both train and validation. We rebuilt the protocol:

1. **Manifest** (`scripts/make_manifest.py`): one row per face crop with its video key, recovered from the crop filename.
2. **Video- and group-level split** (`scripts/make_split_v2.py` → `splits/v2.csv`): ~71% train / 14% val / 14% test, stratified by class. Related FF++ clips (e.g. `008_990` and the source `008`) are linked, and FakeAVCeleb clips are grouped by speaker identity (`idXXXXX`), using union-find over identity tokens. Celeb-DF is grouped by video only, so some identity overlap may remain there.
3. **Leakage check** (`scripts/check_leakage.py`): asserts zero overlap of video keys and groups between train, val and test.
4. **Training** (`training/train_video_v3.py`): sampler weights computed from the train split only, checkpoint chosen by validation loss, test set evaluated once at the end.
5. **Video-level metrics**: crop probabilities are averaged per video key; accuracy and macro-F1 are reported over videos.

Changes to the data in v2: `voiceclone` was removed from the visual classifier (its video is real; it is an audio property), and 500 real FakeAVCeleb videos (`RealVideo-RealAudio`) were added to the `real` class so that "looks like FakeAVCeleb" is not a proxy for "fake".

Diagnostic scripts in `scripts/`: `test_b_source.py` (accuracy by source dataset), `diagnose_misses.py`, `pair_check.py`, `agg_compare.py`, `crop_size_check.py`, `compression_sweep.py`.

---

## Results

| Evaluation | Accuracy | Macro F1 |
|---|---|---|
| 6-class video model (image-level random split, **leaky**, 3,717 images) | 95.0% | 0.89 |
| 6-class video model (video-level leak-free split, 238 test videos) | 83.6% | 0.72 |
| 5-class video model (video-level leak-free split, 310 test videos) | **91.0%** | 0.81 |
| Video-only on fusion labels (34 videos) | 79.4% | 0.63 |
| Video + audio fusion (34 videos) | 73.5% | 0.69 |
| Audio branch on held-out video audio (41 clips) | **87.8%** | — |

### Evaluation note (leak-free split)

The 95% figure was measured on an image-level random split, so crops from the same video could appear in both train and validation. On the leak-free video-level split the numbers are lower, which we consider the honest baseline. The 6-class and 5-class rows use different class sets, test sets and real data, so they are **not directly comparable**.

### 5-class leak-free results (video level, 310 test videos)

| Class | Test videos | Precision | Recall | F1 |
|---|---|---|---|---|
| ai_video | 88 | 0.966 | 0.966 | 0.966 |
| audio_video_fake | 34 | 0.971 | 1.000 | 0.986 |
| edited | 14 | 0.625 | 0.357 | 0.455 |
| faceswap | 22 | 1.000 | 0.591 | 0.743 |
| real | 152 | 0.873 | 0.954 | 0.912 |

Caveats:

- `edited` (14 videos) and `faceswap` (22 videos) have very small test sets; one video moves a score by 4–7 points.
- Results are from a single seed, with no confidence intervals yet.
- `edited` and `faceswap` are the weak classes: most misses are predicted as `real`.
- The `edited` class mixes Face2Face and NeuralTextures crops under shared video keys (25 of 100 keys contain two crop sets), so its per-video numbers are a blend of both methods.
- Source-wise accuracy (`scripts/test_b_source.py`) shows no obvious FakeAVCeleb shortcut for `real` (98.6% on FakeAVCeleb real vs 92.6% on the remaining real clips), but `audio_video_fake` (100%) has not been verified on an external dataset and its crops are noticeably darker than other classes. Treat it as unverified.

### Compression robustness (crop-level degradation, 5-class model)

| Condition | Video acc | Macro F1 | Real recall |
|---|---|---|---|
| clean | 0.910 | 0.812 | 0.954 |
| JPEG 90 | 0.913 | 0.809 | 0.961 |
| JPEG 70 | 0.906 | 0.764 | 0.974 |
| JPEG 50 | 0.903 | 0.761 | 0.967 |
| JPEG 30 | 0.900 | 0.753 | 0.954 |
| downscale 0.5 + JPEG 50 | 0.868 | 0.714 | 0.934 |

This degrades already-extracted crops; it is not a real platform re-encode (H.264, bitrate, frame rate). Per-class scores and the confusion matrix for the original 6-class model are rendered live on the **Model Metrics** page.

---

## Known Limitation

The original 6-class model was validated at 95% accuracy on a leaky split drawn from the same distribution it was trained on (FaceForensics++, Celeb-DF, FakeAVCeleb — landscape, medium-shot, c40-compressed footage). Even on a leak-free split, `edited` and `faceswap` remain weak, and on in-the-wild vertical mobile video the model can label manipulated content as `real` with high confidence. This is a distribution-shift problem, not a code defect. See [Section 6, PROJECT_DOCUMENTATION.md](./PROJECT_DOCUMENTATION.md#6-known-limitation-cross-dataset-generalization-gap).

A `REAL` verdict means **"not detected"** — never proof of authenticity.

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
│   ├── train_video.py       #   original 6-class (leaky split)
│   └── train_video_v3.py    #   5-class on the leak-free split
├── scripts/                 # dataset build, split, leakage check, diagnostics
├── splits/                  # manifest.csv, v1.csv, v2.csv (committed)
├── utils/                   # dataset summary helper
│
├── results/                 # evaluation reports (committed)
│   ├── video_evaluation.txt
│   ├── 6class_video_confusion_matrix.txt
│   ├── v2_test_metrics.json
│   ├── v3_test_metrics.json
│   └── v3_compression_sweep.json
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

## Roadmap

- Cross-dataset evaluation on an in-the-wild benchmark (evaluation only, never used for training)
- Multi-seed runs and bootstrap confidence intervals
- Method-wise `edited` split (Face2Face vs NeuralTextures) and more frames per video for weak classes
- Real fusion training on labeled paired audio-video data, with audio-visual sync features
- Calibration and an "uncertain" output for out-of-distribution video

---

## Team

Maintained by the Multimodal Deepfake Detection team.