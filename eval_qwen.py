"""Correct the saved LoRA drafts with local Qwen2.5-7B-Instruct in 4-bit.

References are used only to compute MER. They are not shown to the model.
"""

import argparse
import json

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from correct_sample import SYSTEM_INSTRUCTION
from mce_common import RESULTS_DIR, mixed_error_rate

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
PAIRS_PATH = RESULTS_DIR / "lora_prompt_pairs.jsonl"
CONSERVATIVE_INSTRUCTION = """
你正在檢查一句語音識別草稿。草稿多數已經正確。
規則：
1. 句子讀起來通順時，原樣輸出，不要改任何字。
2. 不要把廣東話口語改成書面語。保留咁、嘅、咗、哋、喺、嘢、我哋、係、唔。
3. 不要把一個正常詞換成另一個正常詞。不要把燒臘、燒辣、燒鵝、燒鴨互換，不要把「太」改成「真的」或「很」。
4. 英文保持原樣，不要翻譯。
5. 只有在字明顯不成詞時才改正。
6. 只輸出一行最終文本，不要解釋。
"""


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


def first_line(text):
    for line in text.strip().splitlines():
        line = line.strip().strip("\"'")
        if line:
            return line
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", choices=("gemini", "conservative"), default="conservative")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    instruction = SYSTEM_INSTRUCTION if args.prompt == "gemini" else CONSERVATIVE_INSTRUCTION
    suffix = "" if args.prompt == "gemini" else "_conservative"
    progress_path = RESULTS_DIR / f"lora_qwen{suffix}_progress.jsonl"
    output_path = RESULTS_DIR / f"lora_qwen{suffix}.json"

    pairs = load_jsonl(PAIRS_PATH)
    if len(pairs) != 1405:
        raise SystemExit(f"Expected 1405 LoRA drafts, found {len(pairs)}")
    if args.limit:
        pairs = pairs[: args.limit]
    done = {row["index"]: row for row in load_jsonl(progress_path)}
    print(f"Qwen progress {len(done)}/{len(pairs)} prompt {args.prompt}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        ),
        device_map="auto",
    )
    model.eval()

    for pair in tqdm(pairs):
        if pair["index"] in done:
            continue
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": pair["hypothesis"]},
        ]
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=False,
        ).to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                input_ids,
                max_new_tokens=192,
                do_sample=False,
            )
        raw = tokenizer.decode(generated[0][input_ids.shape[-1] :], skip_special_tokens=True)
        row = {
            "index": pair["index"],
            "reference": pair["reference"],
            "hypothesis": pair["hypothesis"],
            "qwen": first_line(raw),
            "qwen_raw": raw.strip(),
            "model": MODEL_NAME,
        }
        append_jsonl(progress_path, row)
        done[pair["index"]] = row
        if args.limit:
            print(f"[{pair['index']}]")
            print(f"Reference: {pair['reference']}")
            print(f"LoRA:      {pair['hypothesis']}")
            print(f"Qwen:      {row['qwen']}")
            print()
        elif len(done) % 25 == 0:
            mer = mixed_error_rate(
                (item["reference"], item["qwen"]) for item in done.values()
            )
            print(f"{len(done)}/{len(pairs)} partial Qwen MER {mer:.2%}", flush=True)

    ordered = [done[index] for index in range(len(pairs))]
    qwen_mer = mixed_error_rate((row["reference"], row["qwen"]) for row in ordered)
    lora_mer = mixed_error_rate((row["reference"], row["hypothesis"]) for row in ordered)
    baseline_mer = json.loads((RESULTS_DIR / "baseline.json").read_text(encoding="utf-8"))["mer"]
    payload = {
        "model": MODEL_NAME,
        "quantization": "4-bit nf4",
        "prompt": args.prompt,
        "sentences": len(ordered),
        "baseline_mer": baseline_mer,
        "lora_mer": lora_mer,
        "qwen_mer": qwen_mer,
        "note": "References were not shown to Qwen.",
        "examples": [
            {
                "reference": row["reference"],
                "lora": row["hypothesis"],
                "qwen": row["qwen"],
            }
            for row in ordered[:20]
        ],
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Baseline MER = {baseline_mer:.2%}")
    print(f"LoRA MER     = {lora_mer:.2%}")
    print(f"Qwen MER     = {qwen_mer:.2%}")
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
