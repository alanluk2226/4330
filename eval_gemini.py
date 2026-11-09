"""Score Gemini corrections on the 1405 LoRA-prompt drafts.

References are saved for MER only. They are not sent to the API.
Progress is appended after every sentence, so a rate limit can be resumed.
"""

import json
import os
import time

import torch
from google import genai
from google.genai import errors, types
from peft import PeftModel
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from correct_sample import SYSTEM_INSTRUCTION, load_env
from mce_common import (
    LANGUAGE,
    MODEL_NAME,
    RESULTS_DIR,
    SPLIT_PATH,
    TASK,
    Corpus,
    load_audio,
    mixed_error_rate,
)

ADAPTER_DIR = "outputs/lora"
PROMPT = "天氣 霧 颱風 能見度 食物 交通 購物 學習 電影"
PAIRS_PATH = RESULTS_DIR / "lora_prompt_pairs.jsonl"
PROGRESS_PATH = RESULTS_DIR / "lora_gemini_progress.jsonl"
OUTPUT_PATH = RESULTS_DIR / "lora_gemini.json"
MODEL = "gemini-3.5-flash"


def load_jsonl(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def append_jsonl(path, row):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_pairs():
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    indices = split["test_indices"]
    print("Loading MCE.", flush=True)
    corpus = Corpus()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    model = PeftModel.from_pretrained(
        WhisperForConditionalGeneration.from_pretrained(MODEL_NAME),
        ADAPTER_DIR,
    ).to(device).eval()
    prompt_ids = processor.get_prompt_ids(PROMPT, return_tensors="pt").to(device)
    RESULTS_DIR.mkdir(exist_ok=True)
    if PAIRS_PATH.exists():
        PAIRS_PATH.unlink()
    batch_size = 8
    written = 0
    for start in tqdm(range(0, len(indices), batch_size)):
        batch_ids = indices[start : start + batch_size]
        audios = []
        references = []
        for index in batch_ids:
            audio_bytes, sentence = corpus.get(index)
            audios.append(load_audio(audio_bytes))
            references.append(sentence)
        features = processor.feature_extractor(
            audios, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device)
        with torch.inference_mode():
            predicted = model.generate(
                features,
                language=LANGUAGE,
                task=TASK,
                prompt_ids=prompt_ids,
            )
        hypotheses = processor.batch_decode(predicted, skip_special_tokens=True)
        for reference, hypothesis in zip(references, hypotheses):
            append_jsonl(
                PAIRS_PATH,
                {"index": written, "reference": reference, "hypothesis": hypothesis},
            )
            written += 1
    print(f"Saved {written} LoRA drafts to {PAIRS_PATH}", flush=True)


def correct(client, draft):
    delay = 2
    last_error = None
    for _ in range(8):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=draft,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=0.1,
                ),
            )
            text = (response.text or "").strip()
            if text:
                return text
            last_error = RuntimeError("empty Gemini response")
        except errors.APIError as exc:
            last_error = exc
            message = str(exc).lower()
            if "per day" in message or "daily" in message:
                raise SystemExit(
                    "Gemini free-tier daily request limit was reached. "
                    "No full-set Gemini MER was computed."
                )
            if getattr(exc, "code", None) not in (429, 503):
                raise
        time.sleep(delay)
        delay = min(delay * 2, 120)
    raise last_error


def main():
    if not PAIRS_PATH.exists():
        write_pairs()
    pairs = load_jsonl(PAIRS_PATH)
    if len(pairs) != 1405:
        raise SystemExit(f"Expected 1405 drafts, found {len(pairs)}")

    done = {row["index"]: row for row in load_jsonl(PROGRESS_PATH)}
    load_env()
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY is missing from .env")
    client = genai.Client(api_key=api_key)
    print(f"Gemini progress {len(done)}/1405", flush=True)

    for pair in pairs:
        if pair["index"] in done:
            continue
        corrected = correct(client, pair["hypothesis"])
        row = {
            "index": pair["index"],
            "reference": pair["reference"],
            "hypothesis": pair["hypothesis"],
            "gemini": corrected,
            "model": MODEL,
        }
        append_jsonl(PROGRESS_PATH, row)
        done[pair["index"]] = row
        if len(done) % 25 == 0:
            mer = mixed_error_rate(
                (item["reference"], item["gemini"]) for item in done.values()
            )
            print(f"{len(done)}/1405 partial Gemini MER {mer:.2%}", flush=True)
        time.sleep(0.4)

    ordered = [done[index] for index in range(1405)]
    gemini_mer = mixed_error_rate((row["reference"], row["gemini"]) for row in ordered)
    lora_mer = mixed_error_rate((row["reference"], row["hypothesis"]) for row in ordered)
    baseline_mer = json.loads((RESULTS_DIR / "baseline.json").read_text(encoding="utf-8"))["mer"]
    payload = {
        "model": MODEL,
        "sentences": 1405,
        "prompt": PROMPT,
        "baseline_mer": baseline_mer,
        "lora_mer": lora_mer,
        "gemini_mer": gemini_mer,
        "note": "References were not sent to Gemini. Partial runs are not this file.",
        "examples": [
            {
                "reference": row["reference"],
                "lora": row["hypothesis"],
                "gemini": row["gemini"],
            }
            for row in ordered[:20]
        ],
    }
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Baseline MER = {baseline_mer:.2%}")
    print(f"LoRA MER     = {lora_mer:.2%}")
    print(f"Gemini MER   = {gemini_mer:.2%}")
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
