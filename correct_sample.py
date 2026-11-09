"""Send LoRA demo transcripts to Gemini and print the corrected text.

The reference is shown only on screen. It is not sent to the API.
"""

import os
import re
from pathlib import Path

from google import genai
from google.genai import errors, types

ROOT = Path(__file__).resolve().parent
TRANSCRIPT = ROOT / "demo_clips" / "transcripts.txt"
MODELS = ("gemini-3.5-flash", "gemini-3.1-flash-lite", "gemini-3.8-flash")

SYSTEM_INSTRUCTION = """
你是一個精通廣東話（粵語）與英語混合輸入的語音識別（ASR）錯字修正引擎。
輸入是由 Whisper + LoRA 識別出來的文本，其中可能包含少量同音字錯別字。

請嚴格遵守以下規則：
1. 只修正明顯的錯別字、同音字誤判，或拼錯的英文字。
2. 必須保留原句的廣東話口語用詞、文法與語氣（如：咁、嘅、囉、係、真係、我哋、咗），不要改成書面語。
3. 英文保持英文，不要翻譯。
4. 如果句子本來就正確，直接回傳原句，不要改寫。
5. 專門注意粵語常見成語、固定詞組與俗語（如：「大義滅親」、「早茶」）。若同音字或近音字破壞了固定詞組（如將「大義滅親」寫成「大意滅親」），修復成正確詞組。訂枱、點心這一類茶樓語境裡，把「早餐」視為「早茶」。
6. 只輸出修正後的最終文本，不要解釋。
"""


def load_env():
    env_path = ROOT / ".env"
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"'))


def load_pairs():
    text = TRANSCRIPT.read_text(encoding="utf-8")
    references = re.findall(r"Reference:\s*(.*)", text)
    drafts = re.findall(r"LoRA:\s*(.*)", text)
    return list(zip(references, drafts))


def correct(client, draft, models=MODELS):
    last_error = None
    for model in models:
        try:
            response = client.models.generate_content(
                model=model,
                contents=draft,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=0.1,
                ),
            )
            return model, (response.text or "").strip()
        except errors.APIError as exc:
            last_error = exc
            if getattr(exc, "code", None) not in (404, 429, 503):
                raise
    raise last_error


def main():
    load_env()
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY is missing from .env")
    client = genai.Client(api_key=api_key)
    for number, (reference, draft) in enumerate(load_pairs(), start=1):
        model, corrected = correct(client, draft)
        print(f"[{number}] {model}")
        print(f"Reference: {reference}")
        print(f"LoRA:      {draft}")
        print(f"Gemini:    {corrected}")
        print()


if __name__ == "__main__":
    main()
