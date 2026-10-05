"""
VISTRA Smart Hat - YAMNet Four-Class Evaluation

This script evaluates YAMNet-based environmental sound recognition
for four categories used in the VISTRA Smart Hat project:

- alarm
- car_horn
- dog_bark
- background

The script performs audio preprocessing, data augmentation,
YAMNet inference, threshold-based classification, and accuracy analysis.

Output files:
- yamnet_fourclass_threshold_010_result.csv
- yamnet_fourclass_threshold_010_summary.csv
- yamnet_fourclass_threshold_010_confusion.csv
"""

import os
import random
import numpy as np
import pandas as pd
import soundfile as sf

from scipy.signal import resample_poly


DATASET_DIR = "dataset"

MODEL_PATH = "yamnet.tflite"
LABEL_PATH = "yamnet_class_map.csv"

TARGET_SR = 16000
TARGET_SAMPLES = 15600

AUGMENT_TIMES = 100

THRESHOLD = 0.10

TRUE_LABELS = [
    "alarm",
    "car_horn",
    "dog_bark",
    "background"
]


CLASS_KEYWORDS = {

    "car_horn": [
        "vehicle horn",
        "car horn",
        "honking",
        "air horn",
        "honk"
    ],

    "dog_bark": [
        "dog",
        "bark",
        "animal"
    ],

    "alarm": [
        "alarm",
        "siren",
        "emergency vehicle",
        "police car",
        "ambulance",
        "fire engine",
        "fire alarm",
        "smoke detector",
        "civil defense siren"
    ]
}


# Manually confirmed windows that should not be counted as alarm
EXCLUDED_ALARM_WINDOWS = {
    ("alarm_005.wav.wav", 4),
    ("alarm_005.wav.wav", 10),
    ("alarm_005.wav.wav", 11),
    ("alarm_010.wav.wav", 29),
    ("alarm_010.wav.wav", 30),
    ("alarm_010.wav.wav", 31),
}


try:
    from tflite_runtime.interpreter import Interpreter
    print("[System] Using tflite_runtime")

except ImportError:
    import tensorflow as tf

    Interpreter = tf.lite.Interpreter
    print("[System] Using TensorFlow Lite Interpreter")


interpreter = Interpreter(
    model_path=MODEL_PATH
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = pd.read_csv(
    LABEL_PATH
)["display_name"].values


def load_audio(path):

    audio, sr = sf.read(path)

    if len(audio.shape) > 1:
        audio = np.mean(
            audio,
            axis=1
        )

    audio = audio.astype(
        np.float32
    )

    if sr != TARGET_SR:

        gcd = np.gcd(
            sr,
            TARGET_SR
        )

        audio = resample_poly(
            audio,
            TARGET_SR // gcd,
            sr // gcd
        ).astype(np.float32)

    max_val = np.max(
        np.abs(audio)
    )

    if max_val > 0:
        audio = audio / max_val

    return audio


def augment_audio(audio):

    x = audio.copy()

    gain = random.uniform(
        0.4,
        1.3
    )

    x *= gain

    noise_level = random.uniform(
        0.001,
        0.035
    )

    x += np.random.normal(
        0,
        noise_level,
        size=len(x)
    )

    shift = random.randint(
        -int(0.2 * TARGET_SR),
        int(0.2 * TARGET_SR)
    )

    x = np.roll(
        x,
        shift
    )

    return np.clip(
        x,
        -1.0,
        1.0
    ).astype(np.float32)


def prepare_input(audio):

    if len(audio) >= TARGET_SAMPLES:

        return audio[
            -TARGET_SAMPLES:
        ].astype(np.float32)

    x = np.zeros(
        TARGET_SAMPLES,
        dtype=np.float32
    )

    x[-len(audio):] = audio

    return x


def run_inference(x):

    interpreter.set_tensor(
        input_details[0]["index"],
        x
    )

    interpreter.invoke()

    return interpreter.get_tensor(
        output_details[0]["index"]
    )[0]


def classify(scores):

    class_scores = {
        "alarm": 0.0,
        "car_horn": 0.0,
        "dog_bark": 0.0
    }

    for custom_label, keywords in CLASS_KEYWORDS.items():

        best = 0.0

        for idx, class_name in enumerate(class_names):

            name = str(
                class_name
            ).lower()

            for keyword in keywords:

                if keyword in name:

                    score = float(
                        scores[idx]
                    )

                    if score > best:
                        best = score

                    break

        class_scores[
            custom_label
        ] = best

    best_label = max(
        class_scores,
        key=class_scores.get
    )

    best_score = class_scores[
        best_label
    ]

    if best_score < THRESHOLD:

        return (
            "background",
            best_score
        )

    return (
        best_label,
        best_score
    )


def test_normal_class(true_label):

    folder = os.path.join(
        DATASET_DIR,
        true_label
    )

    rows = []

    wav_files = sorted([
        f
        for f in os.listdir(folder)
        if f.lower().endswith(".wav")
    ])

    for wav_file in wav_files:

        path = os.path.join(
            folder,
            wav_file
        )

        audio = load_audio(
            path
        )

        for _ in range(
            AUGMENT_TIMES
        ):

            aug = augment_audio(
                audio
            )

            x = prepare_input(
                aug
            )

            scores = run_inference(
                x
            )

            pred, confidence = classify(
                scores
            )

            rows.append({
                "file": wav_file,
                "true_label": true_label,
                "predicted_label": pred,
                "confidence": confidence,
                "correct": int(
                    pred == true_label
                )
            })

    return rows


def test_alarm():

    folder = os.path.join(
        DATASET_DIR,
        "alarm"
    )

    rows = []

    wav_files = sorted([
        f
        for f in os.listdir(folder)
        if f.lower().endswith(".wav")
    ])

    for wav_file in wav_files:

        path = os.path.join(
            folder,
            wav_file
        )

        audio = load_audio(
            path
        )

        start = 0
        window_id = 1

        while start < len(audio):

            end = start + TARGET_SAMPLES

            chunk = audio[
                start:end
            ]

            if len(chunk) == 0:
                break

            if (
                wav_file,
                window_id
            ) in EXCLUDED_ALARM_WINDOWS:

                start += TARGET_SAMPLES
                window_id += 1
                continue

            if len(chunk) < TARGET_SAMPLES:

                padded = np.zeros(
                    TARGET_SAMPLES,
                    dtype=np.float32
                )

                padded[:len(chunk)] = chunk
                chunk = padded

            for _ in range(
                AUGMENT_TIMES
            ):

                aug = augment_audio(
                    chunk
                )

                scores = run_inference(
                    aug
                )

                pred, confidence = classify(
                    scores
                )

                rows.append({
                    "file": wav_file,
                    "window_id": window_id,
                    "true_label": "alarm",
                    "predicted_label": pred,
                    "confidence": confidence,
                    "correct": int(
                        pred == "alarm"
                    )
                })

            start += TARGET_SAMPLES
            window_id += 1

    return rows


def main():

    random.seed(42)
    np.random.seed(42)

    results = []

    print(
        "Testing alarm..."
    )

    results += test_alarm()

    for label in [
        "car_horn",
        "dog_bark",
        "background"
    ]:

        print(
            f"Testing {label}..."
        )

        results += test_normal_class(
            label
        )

    df = pd.DataFrame(
        results
    )

    df.to_csv(
        "yamnet_fourclass_threshold_010_result.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("=" * 70)
    print("Four-Class Evaluation Results")
    print("=" * 70)

    summary = (
        df.groupby(
            "true_label"
        )
        .agg(
            test_count=(
                "correct",
                "count"
            ),
            correct_count=(
                "correct",
                "sum"
            ),
            accuracy=(
                "correct",
                "mean"
            )
        )
    )

    summary[
        "accuracy"
    ] *= 100

    print()
    print(summary)

    overall = (
        df["correct"].mean()
        * 100
    )

    print()
    print(
        f"Overall Accuracy = "
        f"{overall:.2f}%"
    )

    confusion = pd.crosstab(
        df["true_label"],
        df["predicted_label"]
    )

    print()
    print(
        "Confusion Table:"
    )

    print(
        confusion
    )

    summary.to_csv(
        "yamnet_fourclass_threshold_010_summary.csv",
        encoding="utf-8-sig"
    )

    confusion.to_csv(
        "yamnet_fourclass_threshold_010_confusion.csv",
        encoding="utf-8-sig"
    )

    print()
    print(
        "Output files:"
    )

    print(
        "yamnet_fourclass_threshold_010_result.csv"
    )

    print(
        "yamnet_fourclass_threshold_010_summary.csv"
    )

    print(
        "yamnet_fourclass_threshold_010_confusion.csv"
    )


if __name__ == "__main__":
    main()
