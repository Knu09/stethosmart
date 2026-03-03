import os
import librosa
import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, sosfiltfilt
import matplotlib.pyplot as plt
import numpy as np
import cv2
from typing import List, Dict

# Parameters
sr = 22050  # sampling rate
duration = 5.0  # seconds
n_mels = 128
n_mfcc = 40
hop_length = 512

# Label encoding
label_map = {
    "healthy": 0,
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

    print("ICBHI train and test split completed.\n")
    return train_subjs, test_subjs


# -----------------------
# Load Diagnosis Labels
# -----------------------
def load_icbhi_labels(label_file: str):

    print("Loading ICBHI labels split...")
    diag_map = {}
    with open(label_file) as f:
        for line in f:
            print(f"Splitting: {line}")
            parts = line.strip().split()
            if len(parts) == 2:
                subj, diag = parts
                diag_map[subj] = diag.lower()

    print("ICBHI labels split completed.\n")
    return diag_map


# -----------------------
# Build File List using those Splits
# -----------------------
def build_file_list(root, train_subjs, test_subjs, diag_map):
    train_files = []
    test_files = []

    print("Building file list for paths and labels...")
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

    print("Build file list for train and test completed.\n")
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
    print(f"Load audio: \tPATH: {wav_path}\tSR: {sr}")

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
            print(f"Applied bandpass to file: {line}")

            # Normalize amplitude safely
            max_val = np.max(np.abs(cycle_audio))
            if max_val > 0:
                cycle_audio = cycle_audio / max_val
            print(f"Normalize the applitude to file: {line}")

            segments.append(cycle_audio.astype(np.float32))

    print(f"Segment length: {len(segments)}")
    print("Segment cycles completed.\n")
    return segments


# -----------------------
# Build Cycle-Level Dataset
# -----------------------
def build_cycle_dataset(file_list):
    X = []  # independent variable
    y = []  # dependent variable

    print("Building cycle-level dataset...")
    for wav_path, label_name in file_list:
        print(f"Segmented the cycles of file: {wav_path}")
        cycles = segment_cycles(wav_path)

        # Only include labels in label_map
        if label_name not in label_map:
            print(f"Excluded label: {label_name}")
            continue

        for cycle in cycles:
            X.append(cycle)
            y.append(label_map[label_name])

    print("Cycle-Level build completed.\n")
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


def extract_features(
    y, sr=16000, n_mels=128, n_mfcc=40, hop_length=512, target_width=216
) -> np.ndarray:
    # y, sr = librosa.load(y, sr=sr)

    # Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, hop_length=hop_length
    )
    mel_spec_db = librosa.power_to_db(S=mel_spec)

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)

    # Resize all to (128, 216)
    mel_resized = cv2.resize(mel_spec_db, (target_width, n_mels))
    chroma_resized = cv2.resize(chroma, (target_width, n_mels))
    mfcc_resize = cv2.resize(mfcc, (target_width, n_mels))

    # print(f"Mel Spec DB Shape: {mel_resized.shape}")
    # print(f"Chroma Shape: {chroma_resized.shape}")
    # print(f"MFCC Shape: {mfcc_resize.shape}")

    # Resize or pad all features
    # (ensure same shape for all e.g. (3,128,216))
    stacked = np.stack([mel_resized, chroma_resized, mfcc_resize], axis=0).astype(
        np.float32
    )

    # print(f"Final Shape: {stacked.shape}")
    # print(f"Feature Stacked: {stacked}")
    return stacked


# # Sample extraction
# features = extract_features(
#     "../../dataset/ICBHI_final_database/104_1b1_Al_sc_Litt3200.wav"
# )
# print(features)
# plot_stacked_feature(features)
