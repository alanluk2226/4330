"""Transcribe five MCE clips with the original Whisper-small model."""

import io

import pyarrow.parquet as pq
import soundfile as sf
import torch
from transformers import pipeline

table = pq.read_table(r"data/mce/data/train-00000-of-00006.parquet").slice(0, 5)
device = 0 if torch.cuda.is_available() else -1
asr = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    device=device,
)
language = "zh"
print(f"device={device} language={language}")

for i in range(table.num_rows):
    reference = table.column("sentence")[i].as_py()
    audio_bytes = table.column("audio")[i].as_py()["bytes"]
    audio, sampling_rate = sf.read(io.BytesIO(audio_bytes))
    hypothesis = asr(
        {"array": audio, "sampling_rate": sampling_rate},
        generate_kwargs={"language": language, "task": "transcribe"},
    )["text"]
    print(f"REF: {reference}")
    print(f"HYP: {hypothesis}")
    print("---")
