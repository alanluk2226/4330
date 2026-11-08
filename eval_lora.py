"""Score a saved LoRA adapter, optionally with a fixed topic prompt."""

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
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

ADAPTER_DIR = "outputs/lora"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default=ADAPTER_DIR)
    parser.add_argument("--output", default="lora.json")
    parser.add_argument(
        "--prompt",
        default="",
        help="Fixed topic words passed to the decoder. Do not include the reference transcript.",
    )
    args = parser.parse_args()
    adapter_dir = args.adapter

    if not SPLIT_PATH.exists():
        raise SystemExit("Run python eval_baseline.py first.")
    if not (Path(adapter_dir) / "adapter_config.json").exists():
        raise SystemExit(
            f"LoRA weights were not found in {adapter_dir}. "
            "Run python train_lora.py and wait until it prints 'LoRA saved to outputs/lora'."
        )

    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    indices = split["test_indices"]
    print("Loading MCE. This can take a moment.", flush=True)
    corpus = Corpus()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"LoRA model, device {device}, {len(indices)} test sentences", flush=True)

    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    base = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    model = PeftModel.from_pretrained(base, adapter_dir).to(device)
    model.eval()
    prompt_ids = None
    if args.prompt:
        prompt_ids = processor.get_prompt_ids(args.prompt, return_tensors="pt").to(device)
        print(f"Prompt: {args.prompt}", flush=True)

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
        generate_kwargs = {"language": LANGUAGE, "task": TASK}
        if prompt_ids is not None:
            generate_kwargs["prompt_ids"] = prompt_ids
        with torch.inference_mode():
            predicted = model.generate(features, **generate_kwargs)
        hypotheses = processor.batch_decode(predicted, skip_special_tokens=True)
        pairs.extend(zip(references, hypotheses))

    mer = mixed_error_rate(pairs)
    baseline_path = RESULTS_DIR / "baseline.json"
    baseline_mer = None
    if baseline_path.exists():
        baseline_mer = json.loads(baseline_path.read_text(encoding="utf-8"))["mer"]

    payload = {
        "model": MODEL_NAME,
        "adapter": adapter_dir,
        "language": LANGUAGE,
        "task": TASK,
        "prompt": args.prompt,
        "sentences": len(pairs),
        "mer": mer,
        "baseline_mer": baseline_mer,
        "examples": [
            {"reference": reference, "hypothesis": hypothesis}
            for reference, hypothesis in pairs[:20]
        ],
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / args.output).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"LoRA MER = {mer:.2%} (higher is worse)")
    if baseline_mer is not None:
        print(f"Baseline MER = {baseline_mer:.2%}")
    print(f"Saved to results/{args.output}")


if __name__ == "__main__":
    main()
