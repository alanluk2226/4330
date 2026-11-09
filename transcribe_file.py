"""Transcribe one local audio file with the trained LoRA adapter."""

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from mce_common import LANGUAGE, MODEL_NAME, TASK, load_audio

ADAPTER_DIR = Path("outputs/lora")
PROMPT = "天氣 霧 颱風 能見度 食物 交通 購物 學習 電影"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", help="Path to a wav or flac file")
    args = parser.parse_args()
    audio_path = Path(args.audio)
    if not audio_path.is_file():
        raise SystemExit(f"Audio file not found: {audio_path}")
    if not (ADAPTER_DIR / "adapter_config.json").is_file():
        raise SystemExit(f"LoRA adapter not found: {ADAPTER_DIR}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    model = PeftModel.from_pretrained(
        WhisperForConditionalGeneration.from_pretrained(MODEL_NAME),
        str(ADAPTER_DIR),
    ).to(device).eval()
    prompt_ids = processor.get_prompt_ids(PROMPT, return_tensors="pt").to(device)
    audio = load_audio(audio_path.read_bytes())
    features = processor.feature_extractor(
        [audio], sampling_rate=16000, return_tensors="pt"
    ).input_features.to(device)
    with torch.inference_mode():
        predicted = model.generate(
            features,
            language=LANGUAGE,
            task=TASK,
            prompt_ids=prompt_ids,
        )
    text = processor.batch_decode(predicted, skip_special_tokens=True)[0]
    print(text)


if __name__ == "__main__":
    main()
