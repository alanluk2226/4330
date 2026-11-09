"""Transcribe five random clips from a small Hong Kong Cantonese set.

The set is kakiso/KarenSo_CantoneseRecordings. It is not a lazy-sound
benchmark and is not the MCE test split.
"""

import argparse
import csv
import json
import os
import random
from pathlib import Path

import soundfile as sf
import torch
from huggingface_hub import hf_hub_download
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from correct_sample import correct, load_env
from google import genai

from mce_common import LANGUAGE, MODEL_NAME, TASK, load_audio, mixed_error_rate

DATASET = "kakiso/KarenSo_CantoneseRecordings"
ADAPTER_DIR = "outputs/lora"
OUT_DIR = "karen_sample"
PROMPT = "天氣 霧 颱風 能見度 食物 交通 購物 學習 電影"
SEED = 42
COUNT = 5


def load_rows():
    metadata_path = hf_hub_download(DATASET, "metadata.csv", repo_type="dataset")
    with open(metadata_path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def transcribe(model, processor, audio, device, prompt_ids=None):
    features = processor.feature_extractor(
        [audio], sampling_rate=16000, return_tensors="pt"
    ).input_features.to(device)
    generate_kwargs = {"language": LANGUAGE, "task": TASK}
    if prompt_ids is not None:
        generate_kwargs["prompt_ids"] = prompt_ids
    with torch.inference_mode():
        predicted = model.generate(features, **generate_kwargs)
    return processor.batch_decode(predicted, skip_special_tokens=True)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--model", default="gemini-3.1-flash-lite")
    args = parser.parse_args()
    load_env()
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY is missing from .env")
    gemini = genai.Client(api_key=api_key)

    rows = load_rows()
    picked = random.Random(args.seed).sample(range(len(rows)), COUNT)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    baseline = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(device).eval()
    lora = PeftModel.from_pretrained(
        WhisperForConditionalGeneration.from_pretrained(MODEL_NAME),
        ADAPTER_DIR,
    ).to(device).eval()
    prompt_ids = processor.get_prompt_ids(PROMPT, return_tensors="pt").to(device)

    out_dir = OUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    examples = []
    print(f"Dataset: {DATASET}")
    print(f"Random seed {args.seed}, rows {picked}")
    print(f"Prompt: {PROMPT}")
    print()
    for number, index in enumerate(picked, start=1):
        item = rows[index]
        audio_path = hf_hub_download(DATASET, item["file_name"], repo_type="dataset")
        audio = load_audio(Path(audio_path).read_bytes())
        reference = item["hon"]
        clip_path = os.path.join(out_dir, f"{number}.wav")
        sf.write(clip_path, audio, 16000)
        baseline_text = transcribe(baseline, processor, audio, device)
        lora_text = transcribe(lora, processor, audio, device, prompt_ids)
        gemini_model, gemini_text = correct(gemini, lora_text, (args.model,))
        examples.append(
            {
                "row": index,
                "reference": reference,
                "romanization": item["rom"],
                "baseline": baseline_text,
                "lora": lora_text,
                "gemini_model": gemini_model,
                "gemini": gemini_text,
                "clip": clip_path,
            }
        )
        print(f"[{number}] row {index}  {clip_path}")
        print(f"Reference: {reference}")
        print(f"Jyutping:  {item['rom']}")
        print(f"Whisper:   {baseline_text}")
        print(f"LoRA:      {lora_text}")
        print(f"Gemini:    {gemini_text}")
        print()

    summary = {
        "dataset": DATASET,
        "seed": args.seed,
        "rows": picked,
        "sentences": COUNT,
        "prompt": PROMPT,
        "baseline_mer": mixed_error_rate((item["reference"], item["baseline"]) for item in examples),
        "lora_mer": mixed_error_rate((item["reference"], item["lora"]) for item in examples),
        "gemini_mer": mixed_error_rate((item["reference"], item["gemini"]) for item in examples),
        "examples": examples,
        "note": "Five-sentence sample only. Not comparable to the MCE 1405-sentence MER.",
    }
    report_path = os.path.join(out_dir, "sample.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(f"Baseline MER on these 5: {summary['baseline_mer'] * 100:.2f}%")
    print(f"LoRA MER on these 5:     {summary['lora_mer'] * 100:.2f}%")
    print(f"Gemini MER on these 5:   {summary['gemini_mer'] * 100:.2f}%")
    print(f"Saved {report_path}")


if __name__ == "__main__":
    main()
