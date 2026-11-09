"""Score 300 saved LoRA drafts with 20 Gemini calls of 15 sentences each.

References are not sent. The same 300-sentence sample is reused.
"""

import json
import re
import time
from datetime import datetime

from google import genai
from google.genai import errors, types

from correct_sample import SYSTEM_INSTRUCTION, load_env
from mce_common import RESULTS_DIR, mixed_error_rate

PAIRS_PATH = RESULTS_DIR / "lora_prompt_pairs.jsonl"
OUT_DIR = RESULTS_DIR / "gemini_300"
SELECTION_PATH = OUT_DIR / "selection.json"
PROGRESS_PATH = OUT_DIR / "progress_batch15.jsonl"
SUMMARY_PATH = OUT_DIR / "summary_batch15.json"
MODEL = "gemini-3.5-flash"
BATCH_SIZE = 15
BATCHES = 20
RESET_AT = datetime.now().astimezone().replace(hour=16, minute=2, second=0, microsecond=0)

FORMAT_INSTRUCTION = """
下面有 15 句草稿。請依原順序修正。
只回傳一個 JSON 陣列，裡面剛好 15 個字串。
不要編號，不要解釋，不要使用 markdown。
"""


def load_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path, row):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def flatten_selection():
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    rows = [row for part in selection["parts"] for row in part]
    if len(rows) != BATCH_SIZE * BATCHES:
        raise SystemExit(f"Expected {BATCH_SIZE * BATCHES} drafts, found {len(rows)}")
    batches = [rows[start : start + BATCH_SIZE] for start in range(0, len(rows), BATCH_SIZE)]
    for number, batch in enumerate(batches, start=1):
        path = OUT_DIR / f"batch_{number:02d}.jsonl"
        path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in batch) + "\n",
            encoding="utf-8",
        )
    return batches


def parse_batch(text):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    data = json.loads(cleaned)
    if not isinstance(data, list) or len(data) != BATCH_SIZE:
        raise ValueError(f"expected {BATCH_SIZE} strings, got {type(data).__name__} {getattr(data, '__len__', lambda: '?')()}")
    return [str(item).strip() for item in data]


def wait_for_reset():
    now = datetime.now().astimezone()
    if now >= RESET_AT:
        return
    seconds = (RESET_AT - now).total_seconds()
    print(f"Waiting until {RESET_AT.strftime('%H:%M')} for the daily quota reset", flush=True)
    time.sleep(seconds + 5)


def write_summary(done):
    ordered = sorted(done, key=lambda row: (row["batch"], row["offset"]))
    payload = {
        "model": MODEL,
        "calls": BATCHES,
        "sentences_per_call": BATCH_SIZE,
        "sentences_completed": len(ordered),
        "sentences_requested": BATCH_SIZE * BATCHES,
        "complete": len(ordered) == BATCH_SIZE * BATCHES,
        "lora_mer": mixed_error_rate((row["reference"], row["hypothesis"]) for row in ordered) if ordered else None,
        "gemini_mer": mixed_error_rate((row["reference"], row["gemini"]) for row in ordered) if ordered else None,
    }
    SUMMARY_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if ordered:
        print(
            f"Completed {len(ordered)}/{BATCH_SIZE * BATCHES}  "
            f"LoRA MER {payload['lora_mer']:.2%}  Gemini MER {payload['gemini_mer']:.2%}",
            flush=True,
        )
    return payload


def main():
    if not SELECTION_PATH.exists():
        raise SystemExit("Missing results/gemini_300/selection.json")
    batches = flatten_selection()
    done_rows = load_jsonl(PROGRESS_PATH)
    done_batches = {row["batch"] for row in done_rows}
    load_env()
    import os

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY is missing from .env")
    wait_for_reset()
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=1)),
    )
    print(f"Gemini batches done {len(done_batches)}/{BATCHES}", flush=True)

    for number, batch in enumerate(batches, start=1):
        if number in done_batches:
            continue
        numbered = "\n".join(f"{offset + 1}. {row['hypothesis']}" for offset, row in enumerate(batch))
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=FORMAT_INSTRUCTION + "\n" + numbered,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=0.1,
                    max_output_tokens=4096,
                ),
            )
        except errors.APIError as exc:
            message = str(exc).lower()
            write_summary(done_rows)
            if "per day" in message or "daily" in message or getattr(exc, "code", None) == 429:
                raise SystemExit(f"{MODEL} request limit reached at batch {number}.")
            raise
        raw = (response.text or "").strip()
        try:
            corrected = parse_batch(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            append_jsonl(
                PROGRESS_PATH,
                {"batch": number, "error": str(exc), "raw": raw},
            )
            print(f"Batch {number} could not be parsed", flush=True)
            continue
        for offset, (row, text) in enumerate(zip(batch, corrected)):
            saved = {
                "batch": number,
                "offset": offset,
                "index": row["index"],
                "reference": row["reference"],
                "hypothesis": row["hypothesis"],
                "gemini": text,
                "model": MODEL,
            }
            append_jsonl(PROGRESS_PATH, saved)
            done_rows.append(saved)
        done_batches.add(number)
        print(f"Batch {number}/{BATCHES} saved", flush=True)

    write_summary([row for row in done_rows if "gemini" in row])


if __name__ == "__main__":
    main()
