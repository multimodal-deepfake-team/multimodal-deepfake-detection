# 🛡️ Multi-Modal Deepfake Detection

A simple AI-powered web application that checks media for signs of fake faces or synthetic voices. It analyzes audio and visual content separately and combines the available scores to show a final result.

## Features

- Upload videos, audio files, and images.
- Analyze audio for signs of synthetic or spoofed speech.
- Detect faces in images and video frames.
- Analyze face crops for possible manipulation.
- Combine audio and visual scores.
- Display `REAL`, `FAKE`, or `UNCERTAIN`.
- View individual scores and download the report when available.

**Note:** This is a prototype. Its predictions can be incorrect and are not proof that media is real or fake.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Streamlit | Web interface |
| PyTorch | Deep-learning support |
| Hugging Face Transformers | Loading pretrained models |
| OpenCV | Face detection and video processing |
| GenD + CLIP-based model | Visual analysis |
| DF Arena audio model | Audio anti-spoofing |
| FFmpeg | Audio extraction and conversion |

## How It Works

1. Upload a supported media file.
2. The audio model checks for signs of spoofed speech.
3. The visual model analyzes detected faces.
4. The application combines the available scores.
5. The dashboard displays the result and individual scores.

## Project Structure

```text
multimodal-deepfake-detection/
├── app.py
├── detect.py
├── visual.py
├── modelling_gend.py
├── settings.py
├── evaluate.py
├── check_setup.py
├── requirements.txt
├── packages.txt
├── Dockerfile
└── haarcascade_frontalface_default.xml
```

- `app.py` — Streamlit interface and results.
- `detect.py` — Audio analysis and score combination.
- `visual.py` — Video frames, face detection, and visual scoring.
- `modelling_gend.py` — Visual model architecture.
- `settings.py` — Model names and decision thresholds.
- `evaluate.py` — Evaluation on labeled media.
- `check_setup.py` — Environment checks.
- `requirements.txt` — Python dependencies.
- `Dockerfile` — Container setup.

## Installation

### 1. Clone the repository

```powershell
git clone https://github.com/abhishekfaltankar/multimodal-deepfake-detection.git
cd multimodal-deepfake-detection
```

### 2. Create a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install imageio-ffmpeg
```

FFmpeg is required for audio extraction. You can install FFmpeg separately and add it to your system PATH instead.

### 4. Run the application

```powershell
python -m streamlit run app.py
```

Open the local URL shown in the terminal, usually `http://localhost:8501`.

The first run may take several minutes because pretrained models may need to be downloaded.

## Supported File Formats

- **Video:** MP4, MOV, WEBM, MKV, AVI
- **Audio:** MP3, WAV, M4A
- **Images:** JPG, JPEG, PNG

The analysis performed depends on the file type, whether a face is detected, and whether the models load successfully.

## Understanding the Results

The default thresholds in `settings.py` are:

- Fake score of **0.60 or higher**: `FAKE`
- Fake score of **0.40 or lower**: `REAL`
- Between these thresholds: `UNCERTAIN`

When both audio and visual scores are available, the current code combines them using equal weights. A high score from either modality can also trigger a `FAKE` result.

A model score is an estimate and is not necessarily a calibrated probability.

## Evaluation

The repository includes `evaluate.py` for testing with labeled media.

Create this folder structure in the project root:

```text
dataset/
├── real/
└── fake/
```

Place real samples in `real/` and manipulated samples in `fake/`. Then run:

```powershell
python evaluate.py
```

Use separate test samples that were not used to tune the thresholds. Only report accuracy, precision, recall, F1-score, or other metrics after running the evaluation.

## Limitations

- The detector can produce false positives and false negatives.
- Noise, compression, lighting, and recording quality can affect results.
- The audio and visual models may disagree.
- If one model fails or no face is detected, only the available analysis may be used.
- Model downloads can require substantial internet bandwidth and disk space.
- Performance on new media may differ from performance on the original model's evaluation data.

Do not use this tool as the only basis for important legal, academic, or employment decisions.

## Guide for Group Members

Recommended reading order:

1. `settings.py`
2. `detect.py`
3. `visual.py`
4. `app.py`
5. `evaluate.py`

Start by understanding the existing code, test the application, and then make small changes.

## Responsible Use

Only analyze media you have permission to use. Do not upload private recordings, passwords, or access tokens to GitHub.

## License

A project license has not yet been specified. Check the licenses of pretrained models and third-party dependencies before redistributing the project.
