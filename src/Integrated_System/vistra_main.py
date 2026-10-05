"""
VISTRA Smart Hat - Integrated System

This program is part of the VISTRA Smart Hat project,
a Raspberry Pi-based assistive system for hearing-impaired users.

Main functions:
- Real-time speech recognition using whisper.cpp
- Traditional Chinese conversion using OpenCC
- Environmental sound recognition using YAMNet
- Sound-source direction estimation (DoA)
- OLED information display
- LED direction indication

Project: VISTRA Smart Hat
"""

import os
import re
import time
import wave
import queue
import tempfile
import subprocess
import threading
import sys

import numpy as np
import pandas as pd
import pyaudio
import tflite_runtime.interpreter as tflite
import usb.core
import RPi.GPIO as GPIO

from opencc import OpenCC
from luma.core.interface.serial import spi
from luma.oled.device import ssd1322
from luma.core.render import canvas
from PIL import ImageFont


# =========================
# ReSpeaker DoA 模組路徑
# =========================

PIXEL_RING_DIR = os.path.expanduser("~/pixel_ring")

if PIXEL_RING_DIR not in sys.path:
    sys.path.insert(0, PIXEL_RING_DIR)

from usb_4_mic_array.tuning import Tuning


# =========================
# OpenCC
# =========================

# 簡體中文轉換為台灣繁體中文
cc = OpenCC("s2twp")


# =========================
# Lock
# =========================

display_lock = threading.Lock()
result_lock = threading.Lock()


# =========================
# OLED
# =========================

serial = spi(
    device=0,
    port=0,
    gpio_DC=25,
    gpio_RST=24
)

device = ssd1322(serial)

FONT_PATH = os.path.expanduser(
    "~/Desktop/NotoSansCJK-Regular.ttc"
)

try:
    font_main = ImageFont.truetype(
        FONT_PATH,
        16
    )

    font_alert = ImageFont.truetype(
        FONT_PATH,
        22
    )

except Exception:
    font_main = ImageFont.load_default()
    font_alert = ImageFont.load_default()


MAX_CHARS_PER_LINE = 13
MAX_LINES = 2


# =========================
# DoA 與四顆方向 LED
# =========================

LED_FL = 17
LED_FR = 27
LED_BL = 22
LED_BR = 23

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

for pin in (
    LED_FL,
    LED_FR,
    LED_BL,
    LED_BR
):
    GPIO.setup(
        pin,
        GPIO.OUT
    )

    GPIO.output(
        pin,
        GPIO.LOW
    )


# =========================
# ReSpeaker DoA 初始化
# =========================

doa_device = usb.core.find(
    idVendor=0x2886,
    idProduct=0x0018
)

if doa_device is None:
    raise RuntimeError(
        "找不到 ReSpeaker USB 麥克風陣列"
    )

mic_tuning = Tuning(
    doa_device
)

last_doa_angle = 0
last_doa_code = "FF"
last_doa_text = "前方"


# =========================
# LED 控制
# =========================

def all_direction_leds_off():

    for pin in (
        LED_FL,
        LED_FR,
        LED_BL,
        LED_BR
    ):
        GPIO.output(
            pin,
            GPIO.LOW
        )


def angle_to_direction(angle):

    if angle >= 337.5 or angle < 22.5:
        return "FF", "前方"

    elif angle < 67.5:
        return "FL", "左前方"

    elif angle < 112.5:
        return "LL", "左方"

    elif angle < 157.5:
        return "BL", "左後方"

    elif angle < 202.5:
        return "BB", "後方"

    elif angle < 247.5:
        return "BR", "右後方"

    elif angle < 292.5:
        return "RR", "右方"

    else:
        return "FR", "右前方"


def show_direction_led(code):

    all_direction_leds_off()

    if code == "FL":

        GPIO.output(
            LED_FL,
            GPIO.HIGH
        )

    elif code == "FR":

        GPIO.output(
            LED_FR,
            GPIO.HIGH
        )

    elif code == "BL":

        GPIO.output(
            LED_BL,
            GPIO.HIGH
        )

    elif code == "BR":

        GPIO.output(
            LED_BR,
            GPIO.HIGH
        )

    elif code == "FF":

        GPIO.output(
            LED_FL,
            GPIO.HIGH
        )

        GPIO.output(
            LED_FR,
            GPIO.HIGH
        )

    elif code == "BB":

        GPIO.output(
            LED_BL,
            GPIO.HIGH
        )

        GPIO.output(
            LED_BR,
            GPIO.HIGH
        )

    elif code == "RR":

        GPIO.output(
            LED_FR,
            GPIO.HIGH
        )

        GPIO.output(
            LED_BR,
            GPIO.HIGH
        )

    elif code == "LL":

        GPIO.output(
            LED_FL,
            GPIO.HIGH
        )

        GPIO.output(
            LED_BL,
            GPIO.HIGH
        )


# =========================
# whisper.cpp
# =========================

WHISPER_BIN = os.path.expanduser(
    "~/whisper.cpp/build/bin/whisper-cli"
)

WHISPER_MODEL = os.path.expanduser(
    "~/whisper.cpp/models/ggml-tiny.bin"
)


# =========================
# YAMNet
# =========================

# YAMNet 模型與標籤檔
# 必須與本程式放在相同資料夾

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "yamnet.tflite"
)

LABEL_PATH = os.path.join(
    BASE_DIR,
    "yamnet_class_map.csv"
)


interpreter = tflite.Interpreter(
    model_path=MODEL_PATH
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = pd.read_csv(
    LABEL_PATH
)["display_name"].values


TARGET_SAMPLES = 15600
YAMNET_THRESHOLD = 0.1


# =========================
# YAMNet 危險聲音類別
# =========================

YAMNET_CLASS_KEYWORDS = {

    "alarm": [
        "alarm",
        "siren",
        "emergency vehicle",
        "police car",
        "ambulance",
        "fire engine",
        "fire alarm",
        "smoke detector",
        "civil defense siren",
    ],

    "car_horn": [
        "vehicle horn",
        "car horn",
        "honking",
        "air horn",
        "honk",
    ],

    "dog_bark": [
        "dog",
        "bark",
        "animal",
    ],
}


YAMNET_ALERT_TEXT = {

    "alarm":
        "偵測到警報聲",

    "car_horn":
        "偵測到車輛喇叭聲",

    "dog_bark":
        "偵測到狗吠聲",
}


audio_buffer = np.zeros(
    TARGET_SAMPLES,
    dtype=np.float32
)


# =========================
# Audio
# =========================

SR = 16000
CHANNELS = 6
CHUNK = 1024

# ReSpeaker PyAudio index
INPUT_DEVICE_INDEX = 1


# =========================
# Whisper 語音分段參數
# =========================

SPEECH_RMS_THRESHOLD = 0.008

SILENCE_SECONDS = 0.5
MIN_SPEECH_SECONDS = 1.0
MAX_SPEECH_SECONDS = 8.0


SILENCE_CHUNKS = max(
    1,
    int(
        SILENCE_SECONDS
        * SR
        / CHUNK
    )
)

MIN_SPEECH_SAMPLES = int(
    MIN_SPEECH_SECONDS
    * SR
)

MAX_SPEECH_SAMPLES = int(
    MAX_SPEECH_SECONDS
    * SR
)


# =========================
# Queue
# =========================

speech_queue = queue.Queue(
    maxsize=1
)

yamnet_queue = queue.Queue(
    maxsize=20
)

whisper_busy = threading.Event()


# =========================
# 系統辨識結果
# =========================

last_speech_text = ""
last_alert_text = ""
is_danger_sound = False


# =========================
# WAV 儲存
# =========================

def save_wav(
    audio,
    path
):

    pcm = np.clip(
        audio * 32767,
        -32768,
        32767
    ).astype(
        np.int16
    )

    with wave.open(
        path,
        "wb"
    ) as f:

        f.setnchannels(
            1
        )

        f.setsampwidth(
            2
        )

        f.setframerate(
            SR
        )

        f.writeframes(
            pcm.tobytes()
        )


# =========================
# Whisper 辨識
# =========================

def run_whisper(
    wav_path
):

    cmd = [

        WHISPER_BIN,

        "-m",
        WHISPER_MODEL,

        "-f",
        wav_path,

        "-l",
        "zh",

        "-t",
        "3",

        "-nt",
        "-np",
    ]

    try:

        r = subprocess.run(

            cmd,

            capture_output=True,

            text=True,

            timeout=60,
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


        # 移除 Whisper 偶爾產生的括號字幕/註記
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


        # 簡體轉台灣繁體
        return cc.convert(
            text
        )

    except Exception as e:

        print(
            "[Whisper錯誤]",
            e
        )

        return ""


# =========================
# OLED 字幕切行
# =========================

def split_text(
    text
):

    text = text[-26:]

    return [

        text[
            i:
            i + MAX_CHARS_PER_LINE
        ]

        for i in range(
            0,
            len(text),
            MAX_CHARS_PER_LINE
        )

    ][:MAX_LINES]


# =========================
# Whisper 執行緒
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

            whisper_busy.clear()


            if (
                temp_path
                and
                os.path.exists(
                    temp_path
                )
            ):

                os.remove(
                    temp_path
                )


            speech_queue.task_done()


# =========================
# YAMNet 執行緒
# =========================

def yamnet_worker():

    global audio_buffer
    global last_alert_text
    global is_danger_sound


    print(
        "[系統] YAMNet執行緒啟動"
    )


    while True:

        data = yamnet_queue.get()


        try:

            if len(data) >= TARGET_SAMPLES:

                audio_buffer[:] = (
                    data[
                        -TARGET_SAMPLES:
                    ]
                )

            else:

                audio_buffer = np.roll(
                    audio_buffer,
                    -len(data)
                )

                audio_buffer[
                    -len(data):
                ] = data


            # 音量太小就跳過
            if (
                np.sqrt(
                    np.mean(
                        audio_buffer ** 2
                    )
                )
                < 0.005
            ):

                continue


            # YAMNet 推論
            interpreter.set_tensor(

                input_details[0][
                    "index"
                ],

                audio_buffer.astype(
                    np.float32
                ),
            )


            interpreter.invoke()


            scores = (
                interpreter
                .get_tensor(
                    output_details[0][
                        "index"
                    ]
                )[0]
            )


            # 三個自訂危險類別
            class_scores = {

                "alarm": 0.0,

                "car_horn": 0.0,

                "dog_bark": 0.0,
            }


            # 掃描全部 YAMNet 類別
            for (
                custom_label,
                keywords
            ) in (
                YAMNET_CLASS_KEYWORDS
                .items()
            ):

                best_score = 0.0


                for (
                    idx,
                    class_name
                ) in enumerate(
                    class_names
                ):

                    label = str(
                        class_name
                    ).lower()

                    score = float(
                        scores[idx]
                    )


                    for keyword in keywords:

                        if keyword in label:

                            if (
                                score
                                > best_score
                            ):

                                best_score = score

                            break


                class_scores[
                    custom_label
                ] = best_score


            # 找最高分危險類別
            predicted_label = max(

                class_scores,

                key=class_scores.get,
            )


            confidence = (
                class_scores[
                    predicted_label
                ]
            )


            # 低於 threshold
            # 視為背景音
            if (
                confidence
                < YAMNET_THRESHOLD
            ):

                with result_lock:

                    is_danger_sound = False

                continue


            # 危險聲音
            alert_text = (

                YAMNET_ALERT_TEXT
                .get(

                    predicted_label,

                    "偵測到危險聲音",
                )
            )


            with result_lock:

                last_alert_text = (
                    f"{last_doa_text}"
                    f"{alert_text}"
                )

                is_danger_sound = True


            print(

                f"【環境音】"

                f"{predicted_label} "

                f"{confidence:.2f}"
            )


        finally:

            yamnet_queue.task_done()


# =========================
# DoA 執行緒
# =========================

def doa_worker():

    global last_doa_angle
    global last_doa_code
    global last_doa_text


    print(
        "[系統] DoA方向執行緒啟動"
    )


    previous_code = None


    while True:

        try:

            angle = (
                mic_tuning.direction
            )


            if angle is None:

                time.sleep(
                    0.4
                )

                continue


            code, text = (
                angle_to_direction(
                    float(angle)
                )
            )


            with result_lock:

                last_doa_angle = (
                    float(angle)
                )

                last_doa_code = code

                last_doa_text = text


            if code != previous_code:

                show_direction_led(
                    code
                )


                print(

                    f"【DoA】角度 "
                    f"{float(angle):.1f}°，"
                    f"方向：{text}"
                )


                previous_code = code


        except Exception as error:

            print(
                f"[DoA錯誤] "
                f"{error}"
            )


        time.sleep(
            0.4
        )


# =========================
# OLED 執行緒
# =========================

def display_worker():

    global is_danger_sound


    last_displayed = ""


    print(
        "[系統] OLED執行緒啟動"
    )


    while True:

        with result_lock:

            speech = (
                last_speech_text
            )

            danger = (
                is_danger_sound
            )

            alert = (
                last_alert_text
            )

            is_danger_sound = False


        # =========================
        # 危險事件優先顯示
        # =========================

        if danger:

            with display_lock:

                with canvas(
                    device
                ) as draw:

                    draw.rectangle(

                        device.bounding_box,

                        outline="white",

                        fill="black",
                    )


                    draw.text(

                        (25, 18),

                        alert,

                        font=font_alert,

                        fill="white",
                    )


            time.sleep(
                1.5
            )

            last_displayed = alert

            continue


        # =========================
        # 一般語音字幕
        # =========================

        if (
            speech
            and
            speech != last_displayed
        ):

            with display_lock:

                with canvas(
                    device
                ) as draw:

                    draw.rectangle(

                        device.bounding_box,

                        outline="white",

                        fill="black",
                    )


                    y = 5


                    for line in (
                        split_text(
                            speech
                        )
                    ):

                        draw.text(

                            (5, y),

                            line,

                            font=font_main,

                            fill="white",
                        )


                        y += 20


            last_displayed = speech


        time.sleep(
            0.05
        )


# =========================
# 檢查必要檔案
# =========================

for path in [

    WHISPER_BIN,

    WHISPER_MODEL,

    MODEL_PATH,

    LABEL_PATH,
]:

    if not os.path.isfile(
        path
    ):

        raise FileNotFoundError(
            path
        )


# =========================
# 開啟 ReSpeaker 音訊輸入
# =========================

p = pyaudio.PyAudio()


stream = p.open(

    format=pyaudio.paInt16,

    channels=CHANNELS,

    rate=SR,

    input=True,

    input_device_index=(
        INPUT_DEVICE_INDEX
    ),

    frames_per_buffer=CHUNK,
)


# =========================
# 啟動各執行緒
# =========================

threading.Thread(

    target=speech_worker,

    daemon=True,

).start()


threading.Thread(

    target=yamnet_worker,

    daemon=True,

).start()


threading.Thread(

    target=display_worker,

    daemon=True,

).start()


threading.Thread(

    target=doa_worker,

    daemon=True,

).start()


print(
    "Whisper.cpp + YAMNet + "
    "DoA + OLED 持續辨識啟動"
)

print(
    "按 Ctrl+C 結束"
)


# =========================
# Whisper 語音分段狀態
# =========================

speech_samples = []

recording = False

silence_count = 0


# =========================
# 主迴圈
# =========================

try:

    while True:

        data = stream.read(

            CHUNK,

            exception_on_overflow=False,
        )


        raw = np.frombuffer(

            data,

            dtype=np.int16,
        )


        # 從 6-channel 音訊
        # 取第一聲道給 Whisper / YAMNet
        mono = raw[
            ::CHANNELS
        ]


        audio = (

            mono.astype(
                np.float32
            )

            / 32768.0
        )


        rms = float(

            np.sqrt(

                np.mean(
                    audio ** 2
                )
            )
        )


        # =========================
        # Whisper 語音分段
        # =========================

        if (
            rms
            >= SPEECH_RMS_THRESHOLD
        ):

            if not recording:

                print(
                    "[語音] 開始收音"
                )


            recording = True

            silence_count = 0


            speech_samples.extend(

                audio.tolist()
            )


        elif recording:

            speech_samples.extend(

                audio.tolist()
            )


            silence_count += 1


            if (

                silence_count
                >= SILENCE_CHUNKS

                or

                len(
                    speech_samples
                )
                >= MAX_SPEECH_SAMPLES
            ):

                sentence = np.asarray(

                    speech_samples,

                    dtype=np.float32,
                )


                speech_samples.clear()

                recording = False

                silence_count = 0


                if (
                    len(sentence)
                    >= MIN_SPEECH_SAMPLES
                ):

                    print(

                        f"[語音] 送入Whisper："

                        f"{len(sentence) / SR:.2f}秒"
                    )


                    # Whisper 還在忙時
                    # 不堆積舊句子
                    if not (
                        whisper_busy
                        .is_set()
                    ):

                        try:

                            speech_queue.put_nowait(

                                sentence
                            )


                        except queue.Full:

                            print(

                                "[警告] "
                                "Whisper忙碌，"
                                "本句略過"
                            )


                    else:

                        print(

                            "[警告] "
                            "Whisper忙碌，"
                            "本句略過"
                        )


        # =========================
        # YAMNet
        # =========================

        try:

            yamnet_queue.put_nowait(

                audio
            )


        except queue.Full:

            pass


# =========================
# Ctrl+C
# =========================

except KeyboardInterrupt:

    print(
        "\n系統關閉中"
    )


# =========================
# 系統清理
# =========================

finally:

    stream.stop_stream()

    stream.close()

    p.terminate()


    try:

        device.clear()

    except Exception:

        pass


    try:

        all_direction_leds_off()

        GPIO.cleanup()

    except Exception:

        pass
