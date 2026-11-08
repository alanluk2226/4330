"""Train a wider LoRA adapter on attention and feed-forward Whisper modules."""

import json
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from mce_common import LANGUAGE, MODEL_NAME, RESULTS_DIR, SPLIT_PATH, TASK, Corpus
from train_lora import DataCollator, SpeechDataset

OUTPUT_DIR = "outputs/lora_wide"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]


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
            "target_modules": TARGET_MODULES,
        }
        (RESULTS_DIR / "lora_wide_train_stats.json").write_text(
            json.dumps(stats, indent=2), encoding="utf-8"
        )
        print(f"Training time: {elapsed / 60:.1f} minutes")
        print(f"Peak GPU memory: {peak_gb:.2f} GB")


def main():
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    print("Loading MCE for a wider LoRA.", flush=True)
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
            target_modules=TARGET_MODULES,
        ),
    )
    trainable, total = model.get_nb_trainable_parameters()
    print(f"Trainable parameters: {trainable} / {total} ({trainable / total:.2%})")
    args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        warmup_steps=50,
        num_train_epochs=1,
        bf16=torch.cuda.is_available(),
        logging_steps=20,
        save_strategy="no",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="none",
        predict_with_generate=False,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=SpeechDataset(corpus, split["train_indices"], processor),
        data_collator=DataCollator(processor),
        callbacks=[TrainStats()],
    )
    print("Wider LoRA training started. The 17.61% model is left unchanged.", flush=True)
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"LoRA saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
