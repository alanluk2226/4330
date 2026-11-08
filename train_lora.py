"""Train a LoRA adapter on Whisper-small attention projections q_proj and v_proj."""

import json
import time

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from mce_common import LANGUAGE, MODEL_NAME, RESULTS_DIR, SPLIT_PATH, TASK, Corpus, load_audio

OUTPUT_DIR = "outputs/lora"


class SpeechDataset(Dataset):
    def __init__(self, corpus, indices, processor):
        self.corpus = corpus
        self.indices = indices
        self.processor = processor

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        audio_bytes, sentence = self.corpus.get(self.indices[item])
        audio = load_audio(audio_bytes)
        features = self.processor.feature_extractor(
            audio, sampling_rate=16000
        ).input_features[0]
        labels = self.processor.tokenizer(sentence).input_ids
        return {"input_features": features, "labels": labels}


class DataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        batch = self.processor.feature_extractor.pad(
            [{"input_features": feature["input_features"]} for feature in features],
            return_tensors="pt",
        )
        labels_batch = self.processor.tokenizer.pad(
            [{"input_ids": feature["labels"]} for feature in features],
            return_tensors="pt",
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


class TrainStats(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.started = time.perf_counter()

    def on_train_end(self, args, state, control, **kwargs):
        elapsed = time.perf_counter() - self.started
        peak_gb = 0.0
        if torch.cuda.is_available():
            peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        RESULTS_DIR.mkdir(exist_ok=True)
        stats = {
            "train_seconds": elapsed,
            "peak_gpu_gb": peak_gb,
            "steps": state.global_step,
        }
        (RESULTS_DIR / "lora_train_stats.json").write_text(
            json.dumps(stats, indent=2), encoding="utf-8"
        )
        print(f"Training time: {elapsed / 60:.1f} minutes")
        print(f"Peak GPU memory: {peak_gb:.2f} GB")


def main():
    if not SPLIT_PATH.exists():
        raise SystemExit("Run python eval_baseline.py first. It creates the train/test split.")

    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    print("Loading MCE. This can take a moment.", flush=True)
    corpus = Corpus()
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    processor.tokenizer.set_prefix_tokens(language=LANGUAGE, task=TASK)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.config.use_cache = False
    model.generation_config.language = LANGUAGE
    model.generation_config.task = TASK
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            target_modules=["q_proj", "v_proj"],
        ),
    )
    trainable, total = model.get_nb_trainable_parameters()
    print(f"Trainable parameters: {trainable} / {total} ({trainable / total:.2%})")

    dataset = SpeechDataset(corpus, split["train_indices"], processor)
    args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        warmup_steps=50,
        num_train_epochs=1,
        bf16=torch.cuda.is_available(),
        logging_steps=20,
        save_strategy="epoch",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="none",
        predict_with_generate=False,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=DataCollator(processor),
        callbacks=[TrainStats()],
    )
    print(
        "Starting LoRA training. Keep this window open. This can take an hour or more.",
        flush=True,
    )
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"LoRA saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
