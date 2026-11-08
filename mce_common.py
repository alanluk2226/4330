"""Shared data loading, text normalization, and mixed error rate (MER)."""

import io
import random
import unicodedata
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf
import torch
import torchaudio

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "mce" / "data"
SPLIT_PATH = ROOT / "data" / "split.json"
RESULTS_DIR = ROOT / "results"
MODEL_NAME = "openai/whisper-small"
LANGUAGE = "zh"
TASK = "transcribe"
SEED = 42
TEST_RATIO = 0.1


def parquet_files():
    files = sorted(DATA_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"Audio files not found: {DATA_DIR}")
    return files


class Corpus:
    def __init__(self):
        self.files = parquet_files()
        self.tables = [pq.read_table(path) for path in self.files]
        self.offsets = []
        total = 0
        for table in self.tables:
            self.offsets.append(total)
            total += table.num_rows
        self.size = total

    def get(self, index):
        for table, offset in zip(self.tables, self.offsets):
            local = index - offset
            if 0 <= local < table.num_rows:
                audio_bytes = table.column("audio")[local].as_py()["bytes"]
                sentence = table.column("sentence")[local].as_py()
                return audio_bytes, sentence
        raise IndexError(index)


def load_audio(audio_bytes):
    audio, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sample_rate != 16000:
        audio = torchaudio.functional.resample(
            torch.from_numpy(audio), sample_rate, 16000
        ).numpy()
    return audio


def make_split(size):
    indices = list(range(size))
    random.Random(SEED).shuffle(indices)
    test_size = int(round(size * TEST_RATIO))
    return {
        "seed": SEED,
        "test_ratio": TEST_RATIO,
        "test_indices": sorted(indices[:test_size]),
        "train_indices": sorted(indices[test_size:]),
    }


def normalize_text(text):
    chars = []
    for char in text.lower():
        category = unicodedata.category(char)
        if category.startswith("P") or category.startswith("S"):
            chars.append(" ")
        else:
            chars.append(char)
    return "".join(chars)


def mixed_tokens(text):
    tokens = []
    word = []

    def flush():
        if word:
            tokens.append("".join(word))
            word.clear()

    for char in normalize_text(text):
        if "\u4e00" <= char <= "\u9fff":
            flush()
            tokens.append(char)
        elif char.isalnum() or char == "'":
            word.append(char)
        else:
            flush()
    flush()
    return tokens


def edit_distance(reference, hypothesis):
    if not reference:
        return len(hypothesis)
    previous = list(range(len(hypothesis) + 1))
    for i, ref_token in enumerate(reference, start=1):
        current = [i]
        for j, hyp_token in enumerate(hypothesis, start=1):
            cost = 0 if ref_token == hyp_token else 1
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + cost,
                )
            )
        previous = current
    return previous[-1]


def mixed_error_rate(pairs):
    edits = 0
    total = 0
    for reference, hypothesis in pairs:
        ref_tokens = mixed_tokens(reference)
        hyp_tokens = mixed_tokens(hypothesis)
        edits += edit_distance(ref_tokens, hyp_tokens)
        total += len(ref_tokens)
    if total == 0:
        return 0.0
    return edits / total
