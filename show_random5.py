"""Print five random sentences from the 1,215-sentence comparison.

Whisper is transcribed again. LoRA, Qwen, and DeepSeek come from the saved
results for the same sentences. References are not sent to a model.
"""

import argparse
import json
import random

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from mce_common import LANGUAGE, MODEL_NAME, RESULTS_DIR, SPLIT_PATH, TASK, Corpus, load_audio
from score_no_simplified import has_simplified


def load_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--count", type=int, default=5)
    args = parser.parse_args()

    deepseek_rows = [
        row
        for row in load_jsonl(RESULTS_DIR / "lora_deepseek_guard_nosimp_progress.jsonl")
        if not has_simplified(row["reference"])
    ]
    qwen_rows = {row["index"]: row for row in load_jsonl(RESULTS_DIR / "lora_qwen_progress.jsonl")}
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    if len(deepseek_rows) < args.count:
        raise SystemExit(f"Only {len(deepseek_rows)} saved sentences are available.")
    sample = random.Random(args.seed).sample(deepseek_rows, args.count)
    sample.sort(key=lambda row: row["index"])

    corpus = Corpus()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(device).eval()
    print(f"seed {args.seed}  sentences {args.count}  device {device}", flush=True)

    for row in sample:
        audio_bytes, reference = corpus.get(split["test_indices"][row["index"]])
        features = processor.feature_extractor(
            [load_audio(audio_bytes)], sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device)
        with torch.inference_mode():
            predicted = model.generate(features, language=LANGUAGE, task=TASK)
        whisper = processor.batch_decode(predicted, skip_special_tokens=True)[0]
        print(f"\n[{row['index']}]")
        print(f"Reference: {reference}")
        print(f"Whisper:   {whisper}")
        print(f"LoRA:      {row['hypothesis']}")
        print(f"Qwen:      {qwen_rows[row['index']]['qwen']}")
        print(f"DeepSeek:  {row['deepseek']}")


if __name__ == "__main__":
    main()
