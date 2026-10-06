from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

ROOT = Path(r"D:\Final\multimodal_deepfake_detection")
INPUT = ROOT / "dataset" / "audio_subset"
OUTPUT = ROOT / "dataset" / "audio_features"

SR = 16000
N_MELS = 64
N_MFCC = 20
N_FFT = 1024
HOP = 256


def extract_features(path):

    audio, sr = librosa.load(path, sr=SR, mono=True)

    # Remove DC offset
    audio = audio - np.mean(audio)

    # Normalize
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak

    # Log-Mel
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP,
        n_mels=N_MELS,
        power=2.0
    )

    logmel = librosa.power_to_db(mel, ref=np.max)

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SR,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP
    )

    # Delta MFCC
    delta = librosa.feature.delta(mfcc)

    # Energy
    rms = librosa.feature.rms(
        y=audio,
        frame_length=N_FFT,
        hop_length=HOP
    )

    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(
        audio,
        frame_length=N_FFT,
        hop_length=HOP
    )

    # Pitch
    f0, voiced_flag, voiced_prob = librosa.pyin(
        audio,
        fmin=50,
        fmax=500,
        sr=SR,
        frame_length=N_FFT,
        hop_length=HOP
    )

    f0 = np.nan_to_num(f0, nan=0.0)

    # Normalize individual features
    def norm(x):
        mean = np.mean(x)
        std = np.std(x) + 1e-8
        return (x - mean) / std

    logmel = norm(logmel)
    mfcc = norm(mfcc)
    delta = norm(delta)
    rms = norm(rms)
    zcr = norm(zcr)
    f0 = norm(f0.reshape(1, -1))

    return {
        "logmel": logmel.astype(np.float32),
        "mfcc": mfcc.astype(np.float32),
        "delta": delta.astype(np.float32),
        "rms": rms.astype(np.float32),
        "zcr": zcr.astype(np.float32),
        "f0": f0.astype(np.float32)
    }


def process_split(split):

    input_split = INPUT / split
    output_split = OUTPUT / split

    for category in ["real", "tts", "vc"]:

        source_dir = input_split / category
        target_dir = output_split / category

        target_dir.mkdir(parents=True, exist_ok=True)

        files = list(source_dir.glob("*.flac"))

        print(f"\n{split.upper()} / {category}: {len(files)} files")

        for audio_file in tqdm(files):

            try:

                features = extract_features(audio_file)

                output_file = target_dir / (
                    audio_file.stem + ".npz"
                )

                np.savez_compressed(
                    output_file,
                    logmel=features["logmel"],
                    mfcc=features["mfcc"],
                    delta=features["delta"],
                    rms=features["rms"],
                    zcr=features["zcr"],
                    f0=features["f0"]
                )

            except Exception as e:

                print(
                    f"\nERROR: {audio_file.name}: {e}"
                )


process_split("train")
process_split("val")

print("\n================================")
print("FEATURE EXTRACTION COMPLETE")
print("================================")
print(f"Output: {OUTPUT}")
