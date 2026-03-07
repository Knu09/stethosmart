import os
import librosa
import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, sosfiltfilt
import matplotlib.pyplot as plt
import numpy as np
import cv2
from typing import List, Dict
import random
from collections import defaultdict

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
# LABELS
# -----------------------
VALID_LABELS = {"healthy", "asthma", "pneumonia", "copd"}


# -----------------------
# KAUH labels map
# -----------------------
LABEL_MAP = {
    "n": "healthy",
    "asthma": "asthma",
    "copd": "copd",
    "pneumonia": "pneumonia",
}


# -------------------------------------------------
# Parse KAUH Diagnosis from filename
# -------------------------------------------------
def parse_data_label(filename: str):
    """
    Example: BP1_Asthma,I E W,P L L,70,M.wav
             101_1b1_Al_sc_Meditron.wav
    """
    name = filename.replace(".wav", "")
    parts = name.split("_")

    if len(parts) < 2:
        return None

    diagnosis_part = parts[1]
    diagnosis = diagnosis_part.split(",")[0].strip().lower()

    mapped_label = LABEL_MAP.get(diagnosis)
    if mapped_label not in VALID_LABELS:
        return None

    return mapped_label


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
# Inject ICBHI Diagnosis
# -----------------------
def inject_icbhi_diagnosis(diag_map: Dict[str, str], dataset_path):
    """
    Example: 101_1b1_Al_sc_Meditron.wav
    """
    for file in os.listdir(dataset_path):
        if not file.endswith(".wav"):
            continue

        parts = file.split("_")
        subject_id = parts[0]

        if subject_id not in diag_map:
            print(f"No diagnosis found for {file}")
            continue

        diagnosis = diag_map[subject_id]

        # Prevent double renaming
        if diagnosis.lower() in file.lower():
            print(f"Skip renaming {file}")
            continue

        new_filename = f"{subject_id}_{diagnosis}_{'_'.join(parts[1:])}"

        old_path = os.path.join(dataset_path, file)
        new_path = os.path.join(dataset_path, new_filename)

        os.rename(old_path, new_path)

        print(f"Renamed: {file} -> {new_filename}")

    print("\nICBHI filename formatting completed.")


# -----------------------
# Load Diagnosis Labels
# -----------------------
def load_icbhi_labels(label_file: str):

    print("Loading ICBHI labels split...")
    diag_map = {}
    with open(label_file) as f:
        for line in f:
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
# Process Dataset
# -----------------------
def process_dataset(root_path, file_label_getter):
    """
    Generic dataset processor.
    file_label_getter(file) -> returns label string or None
    """
    segments_list = []
    labels_list = []

    for file in os.listdir(root_path):
        if not file.endswith(".wav"):
            continue

        label = file_label_getter(file)
        if label not in VALID_LABELS:
            continue

        full_path = os.path.join(root_path, file)

        segments = segmentation(full_path, sr=16000, window_length=5, hop_length=2.5)

        for seg in segments:
            segments_list.append(seg)
            labels_list.append(label)

    return segments_list, labels_list


# -----------------------
# Extract all segments (ICBHI + KAUH)
# -----------------------
def collect_segments(icbhi_root, icbhi_diag_map, kauh_root):
    all_segments = []
    all_labels = []

    print("Processing ICBHI dataset...")

    def icbhi_label_getter(file):
        patient_id = file.split("_")[0]
        return icbhi_diag_map.get(patient_id)

    icbhi_segments, icbhi_labels = process_dataset(icbhi_root, icbhi_label_getter)

    print("Processing KAUH dataset...")

    def kauh_label_getter(file):
        return parse_data_label(file)

    kauh_segments, kauh_labels = process_dataset(kauh_root, kauh_label_getter)

    all_segments = icbhi_segments + kauh_segments
    all_labels = icbhi_labels + kauh_labels

    return all_segments, all_labels


# -----------------------
# Segment Into Cycles (for ICBHI dataset only)
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
# Segment Into Cycles
# -----------------------


def segmentation(
    wav_path: str, sr: int = 16000, window_length: int = 5, hop_length: float = 2.5
) -> List[np.ndarray]:
    """
    Segment a respiratory recording into fixed-length overlapping windows.

    The parameters contains:
        wav_path: path to .wav file
        sr: target sampling rate (default 16kHz)
        window_length: window size in seconds (default 5s)
        hop_length: hop size in seconds (default 2.5s, 50% overlap)

    Returns a list of dictionaries containing:
        List of audio segments (numpy arrays)
    """
    # Load audio and resample
    y, _ = librosa.load(wav_path, sr=sr)

    # Apply bandpass filter
    y = apply_bandpass(y, sr)
    print(f"Applied bandpass to file: {wav_path}")

    # Normalize amplitude safely
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val
    print(f"Normalize the applitude to file: {wav_path}")

    # Convert seconds to samples
    window_samples = int(window_length * sr)
    hop_samples = int(hop_length * sr)

    segments = []

    # Slide window across signal
    for start in range(0, len(y) - window_samples + 1, hop_samples):
        end = start + window_samples
        segment = y[start:end]

        if len(segment) < sr:
            print(f"Segment cycle is less than the sampling rate: {wav_path}")
            continue

        segments.append(segment.astype(np.float32))

    # If recording is shorter than window_length
    if len(y) < window_samples:
        padded = librosa.util.fix_length(y, size=window_samples)
        segments.append(padded)

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

    # # Chroma
    # chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)

    # Resize all to (128, 216)
    mel_resized = cv2.resize(mel_spec_db, (target_width, n_mels))
    # chroma_resized = cv2.resize(chroma, (target_width, n_mels))
    mfcc_resize = cv2.resize(mfcc, (target_width, n_mels))

    # print(f"Mel Spec DB Shape: {mel_resized.shape}")
    # print(f"Chroma Shape: {chroma_resized.shape}")
    # print(f"MFCC Shape: {mfcc_resize.shape}")

    # Resize or pad all features
    # (ensure same shape for all e.g. (3,128,216))
    stacked = np.stack([mel_resized, mfcc_resize], axis=0).astype(np.float32)

    # print(f"Final Shape: {stacked.shape}")
    # print(f"Feature Stacked: {stacked}")
    return stacked


TARGET_PER_CLASS = 1000


# -----------------------
# Mild Undersampling
# -----------------------
def undersample(X, y):
    class_indices = defaultdict(list)

    for i, label in enumerate(y):
        class_indices[label].append(i)

    new_indices = []

    for label, indices in class_indices.items():
        if label == LABEL_MAP["copd"]:
            # randomly pick 2000
            selected = random.sample(indices, TARGET_PER_CLASS)
            new_indices.extend(selected)
        else:
            new_indices.extend(indices)

    random.shuffle(new_indices)

    X_new = [X[i] for i in new_indices]
    y_new = [y[i] for i in new_indices]

    return X_new, y_new


# -----------------------
# Augmentation Functions
# -----------------------
def augment_audio(y, sr=16000):

    choice = random.choice(["stretch", "pitch", "noise", "shift", "volume"])

    if choice == "stretch":
        y = librosa.effects.time_stretch(y, rate=0.9)

    elif choice == "pitch":
        step = random.choice([-1, 1])
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=step)

    elif choice == "noise":
        noise = np.random.normal(0, 0.005, len(y))
        y = y + noise

    elif choice == "shift":
        shift = int(0.5 * sr)
        y = np.roll(y, shift)

    elif choice == "volume":
        scale = random.uniform(0.5, 1.5)
        y = y * scale

    return y.astype(np.float32)


# -----------------------
# Balance by augmentation
# -----------------------
def balance_by_augmentation(X, y):
    class_indices = defaultdict(list)
    for i, label in enumerate(y):
        class_indices[label].append(i)

    X_balanced = list(X)
    y_balanced = list(y)

    for label, indices in class_indices.items():
        current_count = len(indices)

        if current_count >= TARGET_PER_CLASS:
            continue

        needed = TARGET_PER_CLASS - current_count

        print(f"Augmenting label {label} with {needed} samples")

        for _ in range(needed):
            idx = random.choice(indices)
            augmented = augment_audio(X[idx])
            X_balanced.append(augmented)
            y_balanced.append(label)

    return X_balanced, y_balanced
