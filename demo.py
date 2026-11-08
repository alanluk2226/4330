"""Write five demo clips with baseline and prompted LoRA transcripts."""

import json
import os

import soundfile as sf
import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from mce_common import LANGUAGE, MODEL_NAME, SPLIT_PATH, TASK, Corpus, load_audio

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
ADAPTER_DIR = "outputs/lora"
CLIP_DIR = "demo_clips"
PROMPT = "天氣 霧 颱風 能見度 食物 交通 購物 學習 電影"
DEMO_COUNT = 5


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
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    indices = split["test_indices"][:DEMO_COUNT]
    corpus = Corpus()
    os.makedirs(CLIP_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    baseline = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(device).eval()
    lora = PeftModel.from_pretrained(
        WhisperForConditionalGeneration.from_pretrained(MODEL_NAME),
        ADAPTER_DIR,
    ).to(device).eval()
    prompt_ids = processor.get_prompt_ids(PROMPT, return_tensors="pt").to(device)

    lines = [f"Prompt: {PROMPT}", ""]
    print(lines[0])
    for number, index in enumerate(indices, start=1):
        audio_bytes, reference = corpus.get(index)
        audio = load_audio(audio_bytes)
        clip_path = os.path.join(CLIP_DIR, f"{number}.wav")
        sf.write(clip_path, audio, 16000)
        baseline_text = transcribe(baseline, processor, audio, device)
        lora_text = transcribe(lora, processor, audio, device, prompt_ids)
        block = [
            f"[{number}] {clip_path}",
            f"Reference: {reference}",
            f"Baseline:  {baseline_text}",
            f"LoRA:      {lora_text}",
            "",
        ]
        lines.extend(block)
        print("\n".join(block))

    transcript_path = os.path.join(CLIP_DIR, "transcripts.txt")
    with open(transcript_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print(f"Saved {transcript_path}")


if __name__ == "__main__":
    main()
