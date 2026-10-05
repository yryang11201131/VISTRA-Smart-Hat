import os
import re
import time
import wave
import queue
import tempfile
import subprocess
import threading

import numpy as np
from opencc import OpenCC


# =========================
# Whisper 設定
# =========================

WHISPER_BIN = "/home/alichi/whisper.cpp/build/bin/whisper-cli"
WHISPER_MODEL = "/home/alichi/whisper.cpp/models/ggml-tiny.bin"

cc = OpenCC("s2twp")


# =========================
# Audio 設定
# =========================

SR = 16000
CHANNELS = 6
CHUNK = 1024

SPEECH_RMS_THRESHOLD = 0.008

SILENCE_SECONDS = 0.5
MIN_SPEECH_SECONDS = 1.0
MAX_SPEECH_SECONDS = 8.0

SILENCE_CHUNKS = max(
    1,
    int(SILENCE_SECONDS * SR / CHUNK)
)

MIN_SPEECH_SAMPLES = int(
    MIN_SPEECH_SECONDS * SR
)

MAX_SPEECH_SAMPLES = int(
    MAX_SPEECH_SECONDS * SR
)


# =========================
# Whisper Queue
# =========================

speech_queue = queue.Queue(maxsize=1)

# 判斷 Whisper 是否正在辨識
whisper_busy = threading.Event()

last_speech_text = ""

result_lock = threading.Lock()


# =========================
# 儲存 WAV
# =========================

def save_wav(audio, path):

    pcm = np.clip(
        audio * 32767,
        -32768,
        32767
    ).astype(np.int16)

    with wave.open(path, "wb") as f:

        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SR)

        f.writeframes(
            pcm.tobytes()
        )


# =========================
# 執行 whisper.cpp
# =========================

def run_whisper(wav_path):

    cmd = [
        WHISPER_BIN,
        "-m", WHISPER_MODEL,
        "-f", wav_path,
        "-l", "zh",
        "-t", "3",
        "-nt",
        "-np",
    ]

    try:

        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )

        if r.returncode != 0:

            print(
                "[Whisper錯誤]",
                r.stderr.strip()
            )

            return ""

        text = re.sub(
            r"\s+",
            " ",
            r.stdout
        ).strip()

        # 移除括號類額外字幕
        # 例如：
        # (大陸)
        # （音樂）
        # (笑聲)

        text = re.sub(
            r"\([^)]*\)",
            "",
            text
        )

        text = re.sub(
            r"（[^）]*）",
            "",
            text
        )

        text = text.strip()

        # 簡體轉繁體
        return cc.convert(text)

    except Exception as e:

        print(
            "[Whisper錯誤]",
            e
        )

        return ""


# =========================
# Whisper 背景執行緒
# =========================

def speech_worker():

    global last_speech_text

    print(
        "[系統] Whisper執行緒啟動"
    )

    while True:

        audio = speech_queue.get()

        temp_path = None

        try:

            # 告訴系統 Whisper 正在工作
            whisper_busy.set()

            with tempfile.NamedTemporaryFile(
                suffix=".wav",
                delete=False
            ) as f:

                temp_path = f.name

            save_wav(
                audio,
                temp_path
            )

            start = time.perf_counter()

            text = run_whisper(
                temp_path
            )

            elapsed = (
                time.perf_counter()
                - start
            )

            if text:

                print(
                    f"【Whisper】{text}"
                )

                print(
                    f"【辨識時間】"
                    f"{elapsed:.2f} 秒"
                )

                with result_lock:

                    last_speech_text = text

        finally:

            # Whisper 完成
            whisper_busy.clear()

            if (
                temp_path
                and
                os.path.exists(temp_path)
            ):

                os.remove(temp_path)

            speech_queue.task_done()
