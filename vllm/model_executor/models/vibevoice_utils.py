# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import threading
import numpy as np
from subprocess import run
from typing import Optional, Tuple

def load_audio_use_ffmpeg(file: str, resample: bool = False, target_sr: int = 24000):
    if not resample:
        cmd_probe = [
            "ffprobe",
            "-v", "quiet",
            "-show_entries", "stream=sample_rate",
            "-of", "default=noprint_wrappers=1:nokey=1",
            file
        ]
        original_sr = int(run(cmd_probe, capture_output=True, check=True).stdout.decode().strip())
    else:
        original_sr = None

    sr_to_use = target_sr if resample else original_sr

    cmd = [
        "ffmpeg",
        "-loglevel", "error",
        "-nostdin",
        "-threads", "0",
        "-i", file,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr_to_use),
        "-",
    ]

    out = _run_ffmpeg(cmd).stdout
    audio_data = np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

    return audio_data, sr_to_use

def _get_ffmpeg_max_concurrency() -> int:
    v = os.getenv("VIBEVOICE_FFMPEG_MAX_CONCURRENCY", "")
    try:
        n = int(v) if v.strip() else 0
    except Exception:
        n = 0
    return n

_FFMPEG_MAX_CONCURRENCY = _get_ffmpeg_max_concurrency()
_FFMPEG_SEM = threading.Semaphore(_FFMPEG_MAX_CONCURRENCY) if _FFMPEG_MAX_CONCURRENCY > 0 else None

def _run_ffmpeg(cmd: list, *, stdin_bytes: bytes = None):
    if _FFMPEG_SEM is None:
        return run(cmd, capture_output=True, check=True, input=stdin_bytes)
    with _FFMPEG_SEM:
        return run(cmd, capture_output=True, check=True, input=stdin_bytes)

def load_audio_bytes_use_ffmpeg(data: bytes, *, resample: bool = True, target_sr: int = 24000):
    if not resample:
        raise ValueError("load_audio_bytes_use_ffmpeg requires resample=True")

    cmd = [
        "ffmpeg",
        "-loglevel", "error",
        "-threads", "0",
        "-i", "pipe:0",
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(target_sr),
        "-",
    ]
    out = _run_ffmpeg(cmd, stdin_bytes=data).stdout
    audio_data = np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
    return audio_data, target_sr

class AudioNormalizer:
    def __init__(self, target_dB_FS: float = -25, eps: float = 1e-6):
        self.target_dB_FS = target_dB_FS
        self.eps = eps

    def tailor_dB_FS(self, audio: np.ndarray) -> tuple:
        rms = np.sqrt(np.mean(audio**2))
        scalar = 10 ** (self.target_dB_FS / 20) / (rms + self.eps)
        normalized_audio = audio * scalar
        return normalized_audio, rms, scalar

    def avoid_clipping(self, audio: np.ndarray, scalar: Optional[float] = None) -> tuple:
        if scalar is None:
            max_val = np.max(np.abs(audio))
            if max_val > 1.0:
                scalar = max_val + self.eps
            else:
                scalar = 1.0
        return audio / scalar, scalar

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        audio, _, _ = self.tailor_dB_FS(audio)
        audio, _ = self.avoid_clipping(audio)
        return audio
