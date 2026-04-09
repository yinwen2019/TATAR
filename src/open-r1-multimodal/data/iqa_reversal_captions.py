"""
iqa_reversal_captions.py

Use a multimodal model (e.g., Qwen3-VL-32B served by vLLM) to
reverse-engineer a brief technical reasoning (<think>) for a given IQA MOS score.

Output (jsonl) schema per line:
{
  "image": "...",
  "gt_score": <float>,
  "reversal_thinking": "<think>...</think>\n<answer>\nSCORE\n</answer>"
}

Notes:
- Designed for RL stage-1 cold start: very short "thinking".
- Robust normalization: guarantees EXACT <think> + <answer>, and removes any
  accidental <answer> fragments inside <think>.
"""

import os
import json
import re
import base64
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from openai import OpenAI

# ------------------------------------------------------------
# 0) Disable proxies for intranet vLLM
# ------------------------------------------------------------
for k in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
    os.environ.pop(k, None)

# ============================================================
# 1) Config
# ============================================================
VLLM_BASE_URL = "http://192.168.8.250:8000/v1"
MODEL_NAME = "Qwen3-VL-32B"  # must match vLLM --served-model-name

INPUT_PATH = "/chanxueyan/ano_people/datasets/Data-DeQA-Score/KONIQ/metas/train_koniq_7k.json"
IMAGE_DIR = "/chanxueyan/ano_people/datasets/Data-DeQA-Score"
OUTPUT_PATH = "/chanxueyan/ano_people/datasets/Data-DeQA-Score/KONIQ/train_reversal_thinking.jsonl"

# Test control
MAX_SAMPLES: Optional[int] = 5  # set None to run all

MAX_TOKENS = 140  # short output is enough for 2 sentences + tags
TEMPERATURE = 0.3
TOP_P = 0.8

# Word constraints for <think>
THINK_WORD_MIN = 30
THINK_WORD_MAX = 50

# ============================================================
# 2) Prompts
# ============================================================
SYSTEM_PROMPT = """You are an expert in Image Quality Assessment.

You MUST output exactly:
<think>
A brief technical image quality description.
</think>
<answer>
A single numeric score only, exactly the given score.
</answer>

Rules:
- Discuss ONLY technical image quality (no aesthetics).
- Do NOT invent details not supported by the image.
- Do NOT change the given score.
- The MOS number must NOT appear anywhere inside <think>.
- Follow the format above strictly.
"""

# IMPORTANT: Do NOT include literal "<answer>{score}</answer>" in the prompt,
# otherwise the model may echo it inside <think>.
USER_PROMPT_TEMPLATE = """Ground-truth MOS: {score}

Task:
Based on the provided image, write a brief technical image quality description that explains this MOS.

Constraints for <think>:
- Exactly 2 sentences total.
- 30–50 words total.
- Mention 1–2 key quality factors (e.g., sharpness/blur, noise, compression artifacts, exposure, color cast).
- Do NOT include the MOS number anywhere inside <think>.

Output format (strict):
1) <think> ... </think>
2) <answer> ... </answer> (ONLY the exact MOS number)
"""

# ============================================================
# 3) Helpers
# ============================================================
def guess_mime(image_path: str) -> str:
    ext = Path(image_path).suffix.lower()
    if ext in [".jpg", ".jpeg"]:
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    return "image/jpeg"  # fallback


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def coalesce(d: Dict[str, Any], keys: Tuple[str, ...]) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def load_records(path: str):
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []
    data = json.loads(text)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in ["data", "records", "items", "train"]:
            if k in data and isinstance(data[k], list):
                return data[k]
        return [data]
    raise ValueError("Unsupported input JSON format")


def extract_think(text: str) -> str:
    m = re.search(r"<think>\s*(.*?)\s*</think>", text, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def keep_two_sentences(text: str) -> str:
    """
    Soft enforcement: keep first 2 sentences if more are generated.
    This reduces verbose outputs without relying only on word truncation.
    """
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    sents = [s for s in sents if s]
    return " ".join(sents[:2]).strip() if sents else text.strip()


def clamp_think_words(text: str, wmin: int, wmax: int) -> str:
    words = text.split()
    if len(words) <= wmax:
        return text.strip()
    return " ".join(words[:wmax]).strip()


def normalize_output(raw: str, score: Any) -> str:
    """
    Canonicalize to:
    <think>
    ...
    </think>
    <answer>
    SCORE
    </answer>

    Also removes any accidental <answer> fragments inside <think>,
    including dangling '<answer>...' without closing tag.
    """
    score_str = str(score).strip()

    think = extract_think(raw)
    if not think:
        # fallback: remove answer blocks and think tags, keep the rest
        tmp = re.sub(r"<answer\b[^>]*>.*?</answer>", "", raw, flags=re.DOTALL | re.IGNORECASE)
        tmp = re.sub(r"</?think>", "", tmp, flags=re.IGNORECASE)
        think = tmp.strip()

    # remove stray tags inside think
    think = re.sub(r"</?think>", "", think, flags=re.IGNORECASE)

    # remove complete answer blocks
    think = re.sub(r"<answer\b[^>]*>.*?</answer>", "", think, flags=re.DOTALL | re.IGNORECASE)
    # remove dangling answer tags (no closing </answer>)
    think = re.sub(r"<answer\b[^>]*>.*", "", think, flags=re.DOTALL | re.IGNORECASE)
    # remove stray closing tags
    think = re.sub(r"</answer\s*>", "", think, flags=re.IGNORECASE)
    
    # remove tool-call markers (seen in some Qwen/vLLM outputs)
    think = re.sub(r"</?tool_call\b[^>]*>", "", think, flags=re.IGNORECASE).strip()
    think = re.sub(r"</?tool\b[^>]*>", "", think, flags=re.IGNORECASE).strip()
    think = re.sub(r"</?function_call\b[^>]*>", "", think, flags=re.IGNORECASE).strip()

    # if the model outputs a full tool_call block, drop it
    think = re.sub(r"<tool_call\b[^>]*>.*?</tool_call>", "", think, flags=re.DOTALL | re.IGNORECASE).strip()

    think = think.strip()

    # enforce brevity
    think = keep_two_sentences(think)
    think = clamp_think_words(think, THINK_WORD_MIN, THINK_WORD_MAX)

    return f"<think>\n{think}\n</think>\n<answer>{score_str}</answer>"


# ============================================================
# 4) Main
# ============================================================
def main():
    client = OpenAI(base_url=VLLM_BASE_URL, api_key="EMPTY")

    records = load_records(INPUT_PATH)
    if MAX_SAMPLES:
        records = records[:MAX_SAMPLES]
    print(f"Loaded {len(records)} samples for IQA Reversal.")

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
        for idx, rec in enumerate(records):
            img_name = coalesce(rec, ("image", "image_name", "img_name", "file_name", "filename", "name"))
            score = coalesce(rec, ("gt_score", "mos", "MOS", "score", "quality_score"))

            if img_name is None or score is None:
                print(f"[SKIP] Missing image or score at idx={idx}. keys={list(rec.keys())[:12]}")
                continue

            img_path = os.path.join(IMAGE_DIR, str(img_name))
            if not os.path.exists(img_path):
                print(f"[SKIP] Image not found: {img_path}")
                continue

            mime = guess_mime(img_path)
            b64 = encode_image(img_path)
            user_text = USER_PROMPT_TEMPLATE.format(score=score)

            try:
                resp = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": user_text},
                                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                            ],
                        },
                    ],
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    max_tokens=MAX_TOKENS,
                    stop=["</answer>", "<|endoftext|>"],
                    tools=[],
                    tool_choice="none",
                )

                raw = resp.choices[0].message.content.strip()
                final_text = normalize_output(raw, score)

                out_record = {
                    "image": str(img_name),
                    "gt_score": score,
                    "reversal_thinking": final_text,
                }
                fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")

            except Exception as e:
                print(f"[ERROR] idx={idx} image={img_name} err={e}")

            if (idx + 1) % 20 == 0:
                print(f"Processed {idx + 1}/{len(records)}")

    print(f"Finished! Results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
