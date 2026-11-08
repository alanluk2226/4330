"""Continue the saved LoRA adapter for one more epoch at a lower learning rate."""

import json
import time

import torch
from peft import PeftModel
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback, WhisperForConditionalGeneration, WhisperProcessor

from mce_common import LANGUAGE, MODEL_NAME, RESULTS_DIR, SPLIT_PATH, TASK, Corpus
from train_lora import DataCollator, SpeechDataset

OUTPUT_DIR = "outputs/lora_e2"
RESUME_DIR = "outputs/lora"


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
            "base_adapter": RESUME_DIR,
            "learning_rate": 2e-5,
        }
        (RESULTS_DIR / "lora_e2_train_stats.json").write_text(
            json.dumps(stats, indent=2), encoding="utf-8"
        )
        print(f"Training time: {elapsed / 60:.1f} minutes")
        print(f"Peak GPU memory: {peak_gb:.2f} GB")


def main():
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    print("Loading MCE and the saved LoRA for one more epoch.", flush=True)
    corpus = Corpus()
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    processor.tokenizer.set_prefix_tokens(language=LANGUAGE, task=TASK)
    base = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    base.config.use_cache = False
    base.generation_config.language = LANGUAGE
    base.generation_config.task = TASK
    model = PeftModel.from_pretrained(base, RESUME_DIR, is_trainable=True)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"Trainable parameters: {trainable} / {total} ({trainable / total:.2%})")

    args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        warmup_steps=20,
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
    print("Second epoch started. The 17.61% model is left unchanged.", flush=True)
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"LoRA saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
