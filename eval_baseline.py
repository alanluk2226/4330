"""Score the original Whisper-small model on the held-out MCE test split."""

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
    make_split,
    mixed_error_rate,
    load_audio,
)


def load_split(corpus):
    if SPLIT_PATH.exists():
        return json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    split = make_split(corpus.size)
    SPLIT_PATH.write_text(json.dumps(split), encoding="utf-8")
    print(
        f"Split created: {len(split['train_indices'])} train, "
        f"{len(split['test_indices'])} test"
    )
    return split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N test sentences")
    args = parser.parse_args()

    print("Loading MCE. This can take a moment.", flush=True)
    corpus = Corpus()
    split = load_split(corpus)
    indices = split["test_indices"]
    if args.limit:
        indices = indices[: args.limit]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"Baseline: original Whisper-small, device {device}, {len(indices)} test sentences",
        flush=True,
    )
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    pairs = []
    batch_size = 8
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
            predicted = model.generate(features, language=LANGUAGE, task=TASK)
        hypotheses = processor.batch_decode(predicted, skip_special_tokens=True)
        pairs.extend(zip(references, hypotheses))

    mer = mixed_error_rate(pairs)
    RESULTS_DIR.mkdir(exist_ok=True)
    out_name = "baseline_sample.json" if args.limit else "baseline.json"
    payload = {
        "model": MODEL_NAME,
        "adapter": None,
        "language": LANGUAGE,
        "task": TASK,
        "sentences": len(pairs),
        "mer": mer,
        "examples": [
            {"reference": reference, "hypothesis": hypothesis}
            for reference, hypothesis in pairs[:20]
        ],
    }
    (RESULTS_DIR / out_name).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Baseline MER = {mer:.2%} (higher is worse)")
    print(f"Saved to results/{out_name}")


if __name__ == "__main__":
    main()
