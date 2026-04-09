import json
import re
from pathlib import Path
from typing import Dict, Any
from openai import OpenAI
import os

# 清理代理环境
for env_key in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
    os.environ.pop(env_key, None)

# ============================================================
# 1. 配置区
# ============================================================
VLLM_BASE_URL = "http://192.168.8.250:8000/v1"
MODEL_NAME = "DeepSeek-R1-Distill-Qwen-32B"

INPUT_PATH = "/chanxueyan/ano_people/datasets/ArtiMuse-10K-251219/score/train_with_attrs.json"
# 修改输出文件名以区分
OUTPUT_PATH = "/chanxueyan/ano_people/datasets/ArtiMuse-10K-251219/score/train_cot_structured.jsonl"

# 结构化推理通常比叙述性略长，增加 tokens 限制
MAX_TOKENS = 1024 
TEMPERATURE = 0.4  # 降低温度，增加结构的稳定性
TOP_P = 0.9

# ============================================================
# 2. System Prompt (结构化强约束)
# ============================================================
SYSTEM_PROMPT = """You are a professional aesthetic assessment expert specialized in structural reasoning.

Your task is to generate structured training data for a vision-language model that learns expert-level aesthetic reasoning.. 
You MUST organize your thinking process into distinct logical modules.

You MUST output exactly two sections:

<think>
A single valid JSON object ONLY, describing attribute-based reasoning.
</think>

<answer>
A single numeric score only (no words), exactly matching the given aesthetic score.
</answer>

JSON requirements for <think>:
- The JSON must contain ONLY attribute-level reasoning.
- Do NOT include the final score, task name, or score range.
- Do NOT reference training, models, or datasets.

Rules:
- Output must be VALID JSON inside <think>. No markdown, no extra text.
- Ground every claim strictly in the provided attributes; do not hallucinate details.
- Do NOT introduce new attributes or concepts.
- Synthesize and paraphrase. Do not copy any sentence longer than 12 words.
- The reasoning should logically support the given score, but must not state it.
- Total reasoning length: approximately 100–200 words.
"""

# ============================================================
# 3. User Prompt 模板
# ============================================================
USER_PROMPT_TEMPLATE = """Based on the following aesthetic annotations, generate a structured expert reasoning trace.
These aesthetic attributes cover from low-level visual cues to high-level semantics, and to overall judgment. 


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

You need to create a structured JSON object that conforms to the following template based on these attributes.
You MUST output exactly the following format:

<think>
A single valid JSON object with this EXACT structure:

{{
  "attribute_reasoning": [
    {{
      "name": "visual_technical",
      "source_attributes": ["visual_elements", "technical_execution"],
      "analysis": "<40–70 words of reasoning grounded only in these attributes>",
      "impact": "<one of: strong_positive, positive, slightly_positive, neutral, slightly_negative, negative, strong_negative>"
    }},
    {{
      "name": "composition_layout",
      "source_attributes": ["composition_structure", "overall_gestalt"],
      "analysis": "<35–60 words of reasoning grounded only in these attributes>",
      "impact": "<one of the same impact values>"
    }},
    {{
      "name": "semantics_creativity",
      "source_attributes": ["theme_communication", "emotion_viewer_response", "originality_creativity"],
      "analysis": "<35–60 words of reasoning grounded only in these attributes>",
      "impact": "<one of the same impact values>"
    }},
    {{
      "name": "overall_assessment",
      "source_attributes": ["overall_evaluation"],
      "analysis": "<25–45 words summarizing how the attributes jointly influence the evaluation>",
      "impact": "neutral"
    }}
  ]
}}

</think>

<answer>
{score}
</answer>

Rules:
- Output MUST be valid JSON inside <think>. No markdown, no comments, no extra text.
- Do NOT include the final score or any numeric evaluation inside <think>.
- Do NOT introduce attributes or concepts not present in the input.
- Every claim must be supported by the listed source_attributes.
- Paraphrase and synthesize; do not copy sentences longer than 12 words.
- The reasoning must logically support the given score, but must not state it.
"""

# ============================================================
# 4. 工具函数 (逻辑保持不变，确保清洗彻底)
# ============================================================
def load_records(path: str):
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text: return []
    first = text.lstrip()[0]
    if first == "[":
        return json.loads(text)
    elif first == "{":
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if len(lines) > 1 and all(ln.startswith("{") for ln in lines):
            return [json.loads(ln) for ln in lines]
        return [json.loads(text)]
    raise ValueError("Unsupported format.")

def get_attr(rec: Dict[str, Any], key: str) -> str:
    return rec.get("aesthetic_attributes", {}).get(key, "").strip()

def normalize_cot(text: str, score: Any) -> str:
    score_str = str(score).strip()
    # 提取 answer
    ans_blocks = re.findall(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    ans_raw = ans_blocks[-1].strip() if ans_blocks else ""
    n = re.search(r"[-+]?\d+(\.\d+)?", ans_raw)
    final_ans = n.group(0) if n else score_str

    # 提取 think
    think_blocks = re.findall(r"<think>\s*(.*?)\s*</think>", text, flags=re.DOTALL | re.IGNORECASE)
    if think_blocks:
        think = think_blocks[0].strip()
    else:
        tmp = re.sub(r"<answer>.*?</answer>", "", text, flags=re.DOTALL | re.IGNORECASE)
        think = re.sub(r"</?(think|answer)>", "", tmp, flags=re.IGNORECASE).strip()

    # 移除残留标签，保持紧凑
    think = re.sub(r"</?think>", "", think, flags=re.IGNORECASE).strip()
    return f"<think>\n{think}\n</think>\n<answer>{final_ans}</answer>"

# ============================================================
# 5. 主程序
# ============================================================
def main():
    client = OpenAI(base_url=VLLM_BASE_URL, api_key="EMPTY")
    records = load_records(INPUT_PATH)
    records = records[:4]   # ⭐ 只跑前 10 条
    print(f"Loaded {len(records)} samples for Structured CoT generation.")

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

            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    max_tokens=MAX_TOKENS,
                    stop=["</answer>"]
                )

                assistant_text = response.choices[0].message.content.strip()
                if not assistant_text.endswith("</answer>"):
                    assistant_text += "\n</answer>"
                
                final_text = normalize_cot(assistant_text, rec.get("gt_score", ""))

                out_record = {
                    "image": rec.get("image", ""),
                    "gt_score": rec.get("gt_score", ""),
                    "aesthetic_attributes": rec.get("aesthetic_attributes", {}),
                    "cot_reasoning": final_text,
                }
                fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")

            except Exception as e:
                print(f"Error at idx {idx}: {e}")

            if (idx + 1) % 20 == 0:
                print(f"Structured CoT Processed {idx + 1}/{len(records)}")

    print(f"Done. Saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()