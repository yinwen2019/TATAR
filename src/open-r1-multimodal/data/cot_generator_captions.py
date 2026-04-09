"""
cot_generator_captions.py

Generate full-length Chain-of-Thought (<think> + <answer>)
for IAA cold-start SFT using DeepSeek-R1 via vLLM.

Design goals:
- Full expert-level CoT for every sample
- Strict output format
- Prompt-focused, minimal engineering
"""

import json
import re
from pathlib import Path
from typing import Dict, Any

from openai import OpenAI
import os

os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("ALL_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("all_proxy", None)


# ============================================================
# 1. 配置区（你只需要改这里）
# ============================================================

VLLM_BASE_URL = "http://192.168.8.212:8000/v1"
MODEL_NAME = "DeepSeek-R1-Distill-Qwen-32B"

INPUT_PATH = "/chanxueyan/ano_people/datasets/ArtiMuse-10K-251219/score/train_with_attrs.json"      # jsonl 或 json array
OUTPUT_PATH = "/chanxueyan/ano_people/datasets/ArtiMuse-10K-251219/score/train_cot_narrative.jsonl"

MAX_TOKENS = 500
TEMPERATURE = 0.2
TOP_P = 0.9


# ============================================================
# 2. System Prompt（强约束，建议不动）
# ============================================================

SYSTEM_PROMPT = """You are a professional aesthetic assessment expert.

Your task is to generate training chain-of-thought data for a vision-language model that learns expert-level aesthetic reasoning.

You MUST output exactly two sections:

<think>
A expert chain-of-thought reasoning process.
</think>

<answer>A single numeric score only (no words), exactly matching the given aesthetic score.</answer>

You must strictly follow the following rules:
- First reason step-by-step, and wrap the thought process in <think> tags.
- <answer> must contain ONLY the numeric score (e.g., 2.3). No extra text, no units, no punctuation.
- Do NOT introduce visual details not mentioned.
- Do NOT copy any single sentence longer than 15 words from the attributes.
- Do NOT change the given score.
- You must provide both <think> and <answer>.
- The given score should not appear during the thinking process.
"""


# ============================================================
# 3. User Prompt 模板（你最常改的地方）
# ============================================================

USER_PROMPT_TEMPLATE = """Based on the following aesthetic annotations and the final aesthetic score,
please summarize an expert-level thought process.
These aesthetic attributes cover from low-level visual cues to high-level semantics, and to overall judgment. 
You need to organize a logically coherent thinking process based on these attributes in order to support the final rating.

1) visual_elements:
{visual_elements}

2) composition_structure:
{composition_structure}

3) overall_gestalt:
{overall_gestalt}

4) technical_execution:
{technical_execution}

5) theme_communication:
{theme_communication}

6) emotion_viewer_response:
{emotion_viewer_response}

7) originality_creativity:
{originality_creativity}

8) overall_evaluation:
{overall_evaluation}

Final aesthetic score (must remain unchanged):
{score}

Task requirements:
- Keep <think> concise: 100 words total.
- In <think>, produce a coherent expert internal reasoning narrative (no explicit "Step 1/2/3", no numbered steps).
- In <think>, you must reason using the attributes in the exact order above.
- Synthesize and paraphrase; do not copy long phrases verbatim.
- Make the judgment consistent with the given score (scores below 3.0 are clearly low aesthetic quality).
- Do not introduce any visual details that are not supported by the attributes.

Output format (strict):
<think>
... (one or multiple paragraphs)
</think>
<answer>{score}</answer>
"""



# ============================================================
# 4. 工具函数
# ============================================================

def load_records(path: str):
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []

    first = text.lstrip()[0]
    if first == "[":
        data = json.loads(text)
        if isinstance(data, list):
            return data
        raise ValueError("JSON starts with [ but is not a list.")
    elif first == "{":
        # 可能是 jsonl（多行）或单条 dict
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if len(lines) > 1:
            # jsonl: 每行必须以 { 开头
            if all(ln.startswith("{") for ln in lines):
                return [json.loads(ln) for ln in lines]
        # 单条 dict
        data = json.loads(text)
        if isinstance(data, dict):
            # 如果 dict 里有 list 字段也可以自适应
            return [data]
        raise ValueError("Unsupported JSON object format.")
    else:
        raise ValueError("Unsupported input format (not JSON/JSONL).")

def get_attr(rec: Dict[str, Any], key: str) -> str:
    return rec.get("aesthetic_attributes", {}).get(key, "").strip()

def normalize_cot(text: str, score: Any) -> str:
    """
    Force the output to the exact canonical format:

    <think>
    ...
    </think>
    <answer>
    SCORE
    </answer>

    - Removes duplicate/malformed think/answer tags
    - Ensures <answer> is numeric only
    - If think is missing, uses the text with answer segments removed as fallback
    """
    score_str = str(score).strip()

    # 1) 提取 answer 的数字（如果没有则用 score_str）
    ans_blocks = re.findall(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    ans_raw = ans_blocks[-1].strip() if ans_blocks else ""  # 取最后一个 answer 块（更接近最终输出）
    n = re.search(r"[-+]?\d+(\.\d+)?", ans_raw)
    final_ans = n.group(0) if n else score_str

    # 2) 提取 think 内容：优先取第一个 <think>...</think>
    think_blocks = re.findall(r"<think>\s*(.*?)\s*</think>", text, flags=re.DOTALL | re.IGNORECASE)
    if think_blocks:
        think = think_blocks[0].strip()
    else:
        # 没有 think 标签：去掉所有 answer 段 + 去掉残余标签，作为 think
        tmp = re.sub(r"<answer>.*?</answer>", "", text, flags=re.DOTALL | re.IGNORECASE)
        tmp = re.sub(r"</?think>", "", tmp, flags=re.IGNORECASE)
        tmp = re.sub(r"</?answer>", "", tmp, flags=re.IGNORECASE)
        think = tmp.strip()

    # 3) 进一步清理 think 内部可能残留的 think/answer 标签
    think = re.sub(r"</?think>", "", think, flags=re.IGNORECASE).strip()
    think = re.sub(r"<answer>.*?</answer>", "", think, flags=re.DOTALL | re.IGNORECASE).strip()

    # 4) 最终输出统一格式（不额外塞空行）
    return f"<think>\n{think}\n</think>\n<answer>{final_ans}</answer>"

# ============================================================
# 5. 主生成逻辑
# ============================================================

def main():
    client = OpenAI(
        base_url=VLLM_BASE_URL,
        api_key="EMPTY"  # vLLM 不校验
    )

    records = load_records(INPUT_PATH)
    # records = records[:4]   # ⭐ 只跑前 10 条
    print(f"Loaded {len(records)} samples")

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
        for idx, rec in enumerate(records):
            user_prompt = USER_PROMPT_TEMPLATE.format(
                score=rec.get("gt_score", ""),
                visual_elements=get_attr(rec, "visual_elements"),
                composition_structure=get_attr(rec, "composition_structure"),
                overall_gestalt=get_attr(rec, "overall_gestalt"),
                technical_execution=get_attr(rec, "technical_execution"),
                theme_communication=get_attr(rec, "theme_communication"),
                emotion_viewer_response=get_attr(rec, "emotion_viewer_response"),
                originality_creativity=get_attr(rec, "originality_creativity"),
                overall_evaluation=get_attr(rec, "overall_evaluation"),
            )

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_tokens=MAX_TOKENS,
                stop=["</answer>", "<|endoftext|>"]
            )

            assistant_text = response.choices[0].message.content.strip()
            if not assistant_text.endswith("</answer>"):
                assistant_text += "\n</answer>"
            assistant_text = normalize_cot(assistant_text, rec.get("gt_score", ""))

            out_record = {
                "image": rec.get("image", ""),
                "gt_score": rec.get("gt_score", ""),
                "aesthetic_attributes": rec.get("aesthetic_attributes", {}),
                "cot_reasoning": assistant_text,
            }


            fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")

            if (idx + 1) % 20 == 0:
                print(f"Processed {idx + 1}/{len(records)}")

    print("Done.")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
