"""Score the test sentences whose references contain no simplified characters.

The LoRA number is computed from the saved drafts. The original Whisper
number is computed only when --baseline is passed, and it is written to a
new file. results/baseline.json and results/lora_prompt.json are not changed.
"""

import argparse
import json

import torch
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

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

PAIRS_PATH = RESULTS_DIR / "lora_prompt_pairs.jsonl"
SIMPLIFIED = set(
    "马东车门风鱼鸟页贝见龙飞习书国时这们为过对经发体学头来还进动语话说请谢问题电脑视网关开钟钱饭馆医药业产质环历艺术乐运场万与从应当两现样种个亲爱华丽报导边认识议论计划办法实现无论总统领导经济发展关系关于"
) - set("系")


def has_simplified(text):
    return any(char in SIMPLIFIED for char in text)


def load_pairs():
    rows = [
        json.loads(line)
        for line in PAIRS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 1405:
        raise SystemExit(f"Expected 1405 LoRA drafts, found {len(rows)}")
    return rows


def score_lora():
    rows = load_pairs()
    kept = [row for row in rows if not has_simplified(row["reference"])]
    mer = mixed_error_rate((row["reference"], row["hypothesis"]) for row in kept)
    payload = {
        "sentences": len(kept),
        "dropped_simplified": len(rows) - len(kept),
        "lora_mer": mer,
        "note": "Same saved LoRA drafts. Sentences with simplified characters in the reference were removed from the score.",
    }
    output = RESULTS_DIR / "lora_no_simplified.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Sentences kept = {len(kept)}")
    print(f"Sentences dropped = {len(rows) - len(kept)}")
    print(f"LoRA MER = {mer:.2%}")
    print(f"Saved {output}")


def score_baseline():
    rows = load_pairs()
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    if len(split["test_indices"]) != len(rows):
        raise SystemExit("Test split does not match the saved LoRA drafts.")
    kept_ids = [
        split["test_indices"][row["index"]]
        for row in rows
        if not has_simplified(row["reference"])
    ]
    print(f"Loading MCE for {len(kept_ids)} sentences.", flush=True)
    corpus = Corpus()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(device).eval()
    pairs = []
    batch_size = 8
    for start in tqdm(range(0, len(kept_ids), batch_size)):
        batch_ids = kept_ids[start : start + batch_size]
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
            predicted = model.generate(features, language=LANGUAGE, task=TASK)
        hypotheses = processor.batch_decode(predicted, skip_special_tokens=True)
        pairs.extend(zip(references, hypotheses))
    mer = mixed_error_rate(pairs)
    output = RESULTS_DIR / "baseline_no_simplified.json"
    payload = {
        "model": MODEL_NAME,
        "sentences": len(pairs),
        "baseline_mer": mer,
        "examples": [
            {"reference": reference, "hypothesis": hypothesis}
            for reference, hypothesis in pairs[:20]
        ],
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Sentences kept = {len(pairs)}")
    print(f"Whisper MER = {mer:.2%}")
    print(f"Saved {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()
    if args.baseline:
        score_baseline()
    else:
        score_lora()


if __name__ == "__main__":
    main()
