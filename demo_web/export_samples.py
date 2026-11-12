"""Export a few saved test clips for the local demo page.

Whisper is transcribed again with the baseline settings. LoRA and DeepSeek
come from the saved 1,215-sentence results. This does not call an API and
does not overwrite those result files.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from mce_common import LANGUAGE, MODEL_NAME, RESULTS_DIR, ROOT, SPLIT_PATH, TASK, Corpus, load_audio

INDICES = [290, 278, 926]
OUT_DIR = ROOT / "demo_web"
AUDIO_DIR = OUT_DIR / "audio"


def load_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main():
    deepseek = {
        row["index"]: row
        for row in load_jsonl(RESULTS_DIR / "lora_deepseek_guard_nosimp_progress.jsonl")
    }
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    corpus = Corpus()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(device).eval()
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    samples = []
    for index in INDICES:
        row = deepseek[index]
        audio, reference = corpus.get(split["test_indices"][index])
        if reference != row["reference"]:
            raise SystemExit(f"Reference mismatch at test row {index}")
        waveform = load_audio(audio)
        audio_name = f"{index}.wav"
        sf.write(AUDIO_DIR / audio_name, waveform, 16000)
        features = processor.feature_extractor(
            [waveform], sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device)
        with torch.inference_mode():
            predicted = model.generate(features, language=LANGUAGE, task=TASK)
        whisper = processor.batch_decode(predicted, skip_special_tokens=True)[0]
        samples.append(
            {
                "index": index,
                "audio": f"audio/{audio_name}",
                "reference": row["reference"],
                "whisper": whisper,
                "lora": row["hypothesis"],
                "deepseek": row["deepseek"],
            }
        )
        print(f"[{index}] {whisper}", flush=True)
    (OUT_DIR / "samples.json").write_text(
        json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved {len(samples)} clips to {OUT_DIR}")


if __name__ == "__main__":
    main()
