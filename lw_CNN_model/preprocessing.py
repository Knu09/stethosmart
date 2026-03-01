import os
import librosa
import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, sosfiltfilt
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict

# Parameters
sr = 22050  # sampling rate
duration = 5.0  # seconds
n_mels = 128
n_mfcc = 40
hop_length = 512

# Label encoding
label_map = {
    "normal": 0,
    "copd": 1,
    "asthma": 2,
    "pneumonia": 3,
}


# -----------------------
# Parse Official Split
# -----------------------
def load_icbhi_splits(split_file: str):
    train_subjs = []
    test_subjs = []

    with open(split_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                subj, flag = parts
                if flag == "train":
                    train_subjs.append(subj)
                else:
                    test_subjs.append(subj)

    return train_subjs, test_subjs


# -----------------------
# Load Diagnosis Labels
# -----------------------
def load_icbhi_labels(label_file: str):
    diag_map = {}
    with open(label_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                subj, diag = parts
                diag_map[subj] = diag.lower()

    return diag_map


# -----------------------
# Build File List using those Splits
# -----------------------
def build_file_list(root, train_subjs, test_subjs, diag_map):
    train_files = []
    test_files = []

    for file in os.listdir(root):
        if file.endswith(".wav"):
            recording_id = file.replace(".wav", "")
            patient_id = recording_id.split("_")[0]

            label = diag_map.get(patient_id, None)
            if label is None:
                continue

            full_path = os.path.join(root, file)

            if recording_id in train_subjs:
                train_files.append((full_path, label))
            elif recording_id in test_subjs:
                test_files.append((full_path, label))

    return train_files, test_files


# -----------------------
# Segment Into Cycles
# -----------------------
def segment_cycles(wav_path: str, sr: int = 16000) -> List[Dict]:
    """
    Segment a respiratory recording into cycles using annotation file.
    The parameters contains:
        - wav_path: Wave path
        - sr: Sampling rate (for medical respiratory sounds most papers use 16 KHz)
    Returns a list of dictionaries containing:
        - audio segment
        # - crackle label
        # - wheeze label
    """

    txt_path = wav_path.replace(".wav", ".txt")
    y, _ = librosa.load(wav_path, sr=sr)

    segments = []

    with open(txt_path, "r") as f:
        for line in f:
            start, end, crackle, wheeze = line.strip().split()

            start = float(start)
            end = float(end)
            # NOTE: For multi-task learning (advanced)
            # crackle = int(crackle)
            # wheeze = int(wheeze)

            start_sample = int(start * sr)
            end_sample = int(end * sr)

            cycle_audio = y[start_sample:end_sample]

            # Skips below one second cycle
            if len(cycle_audio) < sr:
                continue

            # NOTE: For multi-task learning (advanced)
            # segments.append(
            #     {"audio": cycle_audio, "crackle": crackle, "wheeze": wheeze}
            # )

            # Apply bandpass filter
            cycle_audio = apply_bandpass(cycle_audio, sr)

            # Normalize amplitude safely
            max_val = np.max(np.abs(cycle_audio))
            if max_val > 0:
                cycle_audio = cycle_audio / max_val

            segments.append(cycle_audio.astype(np.float32))
    return segments


# -----------------------
# Build Cycle-Level Dataset
# -----------------------
def build_cycle_dataset(file_list):
    X = []  # independent variable
    y = []  # dependent variable

    for wav_path, label_name in file_list:
        cycles = segment_cycles(wav_path)

        for cycle in cycles:
            X.append(cycle)
            y.append(label_map[label_name])
    return X, y


def apply_bandpass(
    y: NDArray[np.float64],
    sr: int,
    lowcut: float = 100.0,
    highcut: float = 2000.0,
    order: int = 4,
) -> NDArray[np.float64]:
    """
    Stable bandpass filter using second-order sections.
    """

    nyquist = 0.5 * sr

    low = lowcut / nyquist
    high = highcut / nyquist

    if high >= 1.0:
        high = 0.999

    if low <= 0.0:
        low = 0.001

    sos = butter(order, [low, high], btype="band", output="sos")

    filtered = sosfiltfilt(sos, y)

    return filtered.astype(np.float64)


def plot_stacked_feature(stacked: np.ndarray):
    """
    stacked shape: (3, 128, time)
    """

    titles = ["Mel Spectrogram (dB)", "Chroma (Resized)", "MFCC (Resized)"]

    plt.figure(figsize=(12, 8))

    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.imshow(stacked[i], aspect="auto", origin="lower", cmap="magma")
        plt.title(titles[i])
        plt.colorbar()

    plt.tight_layout()
    plt.show()


def resize_feature(feature: np.ndarray, target_height: int) -> np.ndarray:
    """
    Resize feature to (raget_height, time) with simple interpolation.
    """

    return librosa.util.fix_length(
        np.array(
            [
                np.interp(
                    np.linspace(0, feature.shape[0] - 1, target_height),
                    np.arange(feature.shape[0]),
                    feature[:, t],
                )
                for t in range(feature.shape[1])
            ]
        ).T,
        size=feature.shape[1],
        axis=1,
    )


def extract_features(y, sr=16000) -> np.ndarray:
    # pad/trim to fixed length
    y = librosa.util.fix_length(y, size=int(sr * duration))

    # Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, hop_length=hop_length
    )
    mel_spec_db = librosa.power_to_db(S=mel_spec)

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    print(f"Mel Spec DB Shape: {mel_spec_db.shape}")
    print(f"Chroma Shape: {chroma.shape}")
    print(f"MFCC Shape: {mfcc.shape}")

    # Resize to 128 rows
    chroma_resized = resize_feature(chroma, n_mels)
    mfcc_resize = resize_feature(mfcc, n_mels)

    # Resize to same time dimension
    min_time = min(
        mel_spec_db.shape[1],
        chroma.shape[1],
        mfcc.shape[1],
    )

    mel_spec_db = mel_spec_db[:, :min_time]
    chroma = chroma_resized[:, :min_time]
    mfcc = mfcc_resize[:, :min_time]

    # Resize or pad all features
    # (ensure same shape for all, e.g., mel: (128,216), chroma: (12,216), mfcc: (40,216))
    stacked = np.stack([mel_spec_db, chroma_resized, mfcc_resize], axis=0)

    print(f"Final Shape: {stacked.shape}")
    print(f"Feature Stacked: {stacked}")
    return stacked


# # Sample extraction
# features = extract_features(
#     "../../dataset/ICBHI_final_database/102_1b1_Ar_sc_Meditron.wav"
# )
# print(features)
# plot_stacked_feature(features)
