#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iaa_judge.py (minimal)

Input JSONL schema (per line):
{
  "image": "10_193.jpg",
  "gt_score": 2.3,
  "aesthetic_attributes": { ... },   # dict of attribute_name -> long text
  "cot_reasoning": "<think>...</think>\n<answer>2.3</answer>"
}

Output JSONL (kept only):
train_cot_narrative_judged.jsonl
"""

import argparse
import base64
import imghdr
import json
import os
import re
import sys
import requests
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

TAG_THINK = re.compile(r"<think>(.*?)</think>", re.S | re.I)
TAG_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)
NUM_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d+)?|\.\d+)$")

BAD_RE = re.compile(
    r"(<\s*tool_call\s*>|</\s*tool_call\s*>|\bsystem\s*:|\bassistant\s*:|\bdeveloper\s*:|```|<\s*script\b)",
    re.I,
)
NESTED_TAG_RE = re.compile(r"<\s*[a-zA-Z_][\w\-]*\s*>|<\s*/\s*[a-zA-Z_][\w\-]*\s*>")

# --- Attribute weak-matching keywords (tune if needed) ---
ATTR_KEYWORDS = {
    "emotion_viewer_response": ["emotion", "emotional", "mood", "feeling", "evoke", "impact", "resonance", "viewer"],
    "composition_structure": ["composition", "layout", "balance", "symmetry", "framing", "alignment", "spacing", "rule of thirds", "structure"],
    "originality_creativity": ["original", "originality", "creative", "creativity", "novel", "unique", "generic", "derivative", "innovative"],
    "visual_elements": ["color", "palette", "contrast", "saturation", "shape", "line", "texture", "space", "depth", "lighting", "shadow"],
    "overall_gestalt": ["gestalt", "cohesive", "coherence", "harmony", "unity", "holistic", "overall"],
    "technical_execution": ["technical", "execution", "polish", "refined", "resolution", "artifact", "edges", "blending", "rendering", "clarity"],
    "theme_communication": ["theme", "story", "narrative", "message", "communication", "concept", "context", "cultural"],
    "overall_evaluation": ["overall", "aesthetic", "appeal", "quality", "evaluation", "rating", "score"],
}

CONNECTOR_HINTS = ["because", "due to", "therefore", "thus", "so", "as a result", "which makes", "leading to", "hence"]


JUDGE_PROMPT_EN = r"""
You are an extremely strict and conservative aesthetic assessment (IAA) data auditor.

You will receive:
- an image
- gt_score (ground-truth aesthetic score)
- predicted_score (from <answer>)
- aesthetic_attributes: a dictionary of attribute_name -> descriptive text (grounded reference)
- cot_think: the model's reasoning text

Your task: decide whether this sample is suitable for training.

Hard expectations for IAA reasoning quality:
A) Coverage: cot_think should address AT LEAST ~5 attribute aspects (weak semantic match is OK; not strict per-attribute alignment).
B) No fabrication / distortion: Do NOT introduce aesthetic claims that are not supported by the image OR that clearly contradict the provided aesthetic_attributes.
C) Not mere parroting: It must go beyond repeating attributes; it should connect attributes logically (e.g., "soft lighting => enhanced mood").
D) Smooth derivation: The path from visual description -> attribute implications -> final score should be coherent, without abrupt logical leaps.

You MUST:
- Judge ONLY based on the image + aesthetic_attributes. Do NOT invent image details.
- Be conservative: any clear mismatch, fabrication, or shallow/illogical reasoning => reject.
- Output ONLY one strictly valid JSON object (no extra text, no markdown).

Output JSON schema (STRICT):
{
  "decision": "keep" | "reject",
  "overall_confidence": 0-100,
  "scores": {
    "attribute_coverage": 0-5,
    "groundedness_no_fabrication": 0-5,
    "reasoning_cohesion": 0-5,
    "score_derivation_smoothness": 0-5
  },
  "major_issues": ["..."],   // if reject, MUST be non-empty
  "minor_notes": ["..."],
  "debug": {"gt_score":<number>,"predicted_score":<number>,"abs_diff":<number>}
}

Guidance (conservative):
- If groundedness_no_fabrication <=2 OR reasoning_cohesion <=2 => strongly prefer reject.
- If it mostly parrots attributes with no causal/logical linking => prefer reject.
- If it jumps to the score without a coherent chain => prefer reject.
""".strip()


def img_to_data_uri(path: str) -> str:
    kind = imghdr.what(path) or os.path.splitext(path)[1].lower().lstrip(".")
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}.get(kind, f"image/{kind or 'jpeg'}")
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def hard_parse_cot(cot: str):
    if not cot or BAD_RE.search(cot):
        return None, None, None, "contamination_or_empty"

    t = TAG_THINK.findall(cot)
    a = TAG_ANSWER.findall(cot)
    if len(t) != 1 or len(a) != 1:
        return None, None, None, "tag_count_invalid"

    # exactly once each (open/close)
    if len(re.findall(r"<\s*think\s*>", cot, flags=re.I)) != 1 or len(re.findall(r"</\s*think\s*>", cot, flags=re.I)) != 1:
        return None, None, None, "think_tag_not_exactly_once"
    if len(re.findall(r"<\s*answer\s*>", cot, flags=re.I)) != 1 or len(re.findall(r"</\s*answer\s*>", cot, flags=re.I)) != 1:
        return None, None, None, "answer_tag_not_exactly_once"

    think = t[0].strip()
    ans_raw = a[0].strip()

    if not NUM_RE.match(ans_raw):
        return None, None, None, "answer_not_pure_number"
    if NESTED_TAG_RE.search(think) or NESTED_TAG_RE.search(ans_raw):
        return None, None, None, "nested_tags_in_think_or_answer"

    return think, ans_raw, float(ans_raw), "ok"


def count_attribute_aspects(think: str, aesthetic_attributes: dict) -> int:
    """
    Weak matching: count how many attribute categories are "touched" in think.
    Uses keyword hits; no strict alignment to attribute text.
    """
    if not isinstance(aesthetic_attributes, dict):
        return 0
    text = (think or "").lower()

    covered = 0
    for k in aesthetic_attributes.keys():
        kws = ATTR_KEYWORDS.get(k, None)
        if not kws:
            # fallback: try splitting key name
            kws = k.replace("_", " ").split()
        if any(kw.lower() in text for kw in kws):
            covered += 1
    return covered


def call_judge(api_url: str, api_key: str, model: str, image_data_uri: str, user_text: str, timeout: int = 120) -> dict:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": JUDGE_PROMPT_EN},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": image_data_uri}},
                ],
            },
        ],
    }
    r = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"].strip()
    return json.loads(content)  # strict JSON-only expected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/chanxueyan/ano_people/datasets/ArtiMuse-10K-251219/score/train_cot_narrative.jsonl")
    ap.add_argument("--output", default="train_cot_narrative_judged.jsonl")
    ap.add_argument("--image_root", default="/chanxueyan/ano_people/datasets/ArtiMuse-10K-251219/images")
    ap.add_argument("--api_key", default="sk-lui3w1QQftJEKZ51596cA4E603A54398B5C0B44a666dE685")
    ap.add_argument("--api_url", default="http://192.168.9.180:8123/v1/chat/completions")
    ap.add_argument("--model", default="qwen3_VL_235B")
    ap.add_argument("--min_score", type=float, default=1.0)
    ap.add_argument("--max_score", type=float, default=100.0)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--max_lines", type=int, default=0)
    ap.add_argument("--min_aspects", type=int, default=5, help="IAA rule: think must touch >= N attribute aspects")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--print_decisions", action="store_true", help="Print per-sample decision/drop reason in real time")
    ap.add_argument("--num_workers", type=int, default=10, help="Number of worker threads for parallel processing")
    args = ap.parse_args()

    kept = drop_hard = drop_attr = drop_judge = judge_err = total = 0
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    lock = threading.Lock()
    pbar_lock = threading.Lock()

    # Read all lines first so progress_total can default to JSON line count.
    lines = []
    with open(args.input, "r", encoding="utf-8") as fin:
        for idx, line in enumerate(fin):
            if args.max_lines and idx >= args.max_lines:
                break
            line = line.strip()
            if not line:
                continue
            lines.append((len(lines) + 1, line))

    total = len(lines)

    progress_total = args.max_lines if args.max_lines > 0 else total
    pbar = tqdm(total=progress_total, desc="Judging IAA", unit="sample", file=sys.stderr) if tqdm else None
    output_queue = Queue()

    def emit_decision(msg: str):
        if not args.print_decisions:
            return
        with pbar_lock:
            if pbar is not None:
                pbar.write(msg)
            else:
                print(msg, file=sys.stderr)

    def process_sample(item_data):
        line_idx, line = item_data

        try:
            item = json.loads(line)
        except Exception:
            emit_decision(f"[{line_idx}] dec=drop_hard reason=json_parse_error")
            return None, {"drop_hard": 1}

        img_rel = item.get("image")
        gt = item.get("gt_score")
        attrs = item.get("aesthetic_attributes")
        cot = item.get("cot_reasoning") or item.get("reversal_thinking")

        if not isinstance(img_rel, str) or not isinstance(gt, (int, float)) or not isinstance(cot, str) or not isinstance(attrs, dict):
            emit_decision(f"[{line_idx}] dec=drop_hard reason=invalid_fields")
            return None, {"drop_hard": 1}

        think, ans_raw, pred, reason = hard_parse_cot(cot)
        if reason != "ok":
            emit_decision(f"[{line_idx}] dec=drop_hard reason={reason}")
            return None, {"drop_hard": 1}

        if not (args.min_score <= float(gt) <= args.max_score and args.min_score <= float(pred) <= args.max_score):
            emit_decision(
                f"[{line_idx}] dec=drop_hard reason=score_out_of_range "
                f"gt={float(gt):.4f} pred={float(pred):.4f}"
            )
            return None, {"drop_hard": 1}

        # IAA-specific local rule: >=N attribute aspects touched.
        aspects = count_attribute_aspects(think, attrs)
        if aspects < args.min_aspects:
            emit_decision(
                f"[{line_idx}] dec=drop_attr reason=insufficient_attribute_aspects "
                f"aspects={aspects} min_aspects={args.min_aspects}"
            )
            return None, {"drop_attr": 1}

        img_path = img_rel if os.path.isabs(img_rel) else os.path.join(args.image_root, img_rel)
        if not os.path.exists(img_path):
            emit_decision(f"[{line_idx}] dec=drop_hard reason=image_not_found image={img_rel}")
            return None, {"drop_hard": 1}

        # Light heuristic: check if there is any causal linking language (not mandatory, but useful hint).
        has_linking = any(h in (think or "").lower() for h in CONNECTOR_HINTS)

        try:
            image_data_uri = img_to_data_uri(img_path)

            # Keep attributes reasonably bounded (avoid extreme payload size).
            attrs_compact = {k: (v[:1200] + "..." if isinstance(v, str) and len(v) > 1200 else v) for k, v in attrs.items()}

            user_text = (
                f"scale_min: {args.min_score}\nscale_max: {args.max_score}\n"
                f"gt_score: {float(gt)}\npredicted_score: {float(pred)}\n"
                f"local_attribute_aspects_touched: {aspects} (min_required={args.min_aspects})\n"
                f"local_has_causal_linking_terms: {bool(has_linking)}\n"
                f"aesthetic_attributes (reference): {json.dumps(attrs_compact, ensure_ascii=False)}\n"
                f"cot_think:\n{think}\n"
            )
            judge = call_judge(args.api_url, args.api_key, args.model, image_data_uri, user_text, args.timeout)
        except Exception:
            emit_decision(f"[{line_idx}] dec=judge_err reason=api_or_parse_exception image={img_rel}")
            return None, {"judge_err": 1}

        # Minimal validation + conservative enforce.
        if not isinstance(judge, dict):
            emit_decision(f"[{line_idx}] dec=drop_judge reason=judge_not_dict image={img_rel}")
            return None, {"drop_judge": 1}

        dec = judge.get("decision")
        scores = judge.get("scores")
        if not isinstance(scores, dict):
            emit_decision(f"[{line_idx}] dec=drop_judge reason=scores_not_dict image={img_rel}")
            return None, {"drop_judge": 1}
        gnd = scores.get("groundedness_no_fabrication")
        coh = scores.get("reasoning_cohesion")

        if dec not in ("keep", "reject") or not isinstance(gnd, (int, float)) or not isinstance(coh, (int, float)):
            emit_decision(
                f"[{line_idx}] dec=drop_judge reason=judge_schema_invalid "
                f"raw_dec={dec} gnd={gnd} coh={coh} image={img_rel}"
            )
            return None, {"drop_judge": 1}

        raw_dec = dec
        if gnd <= 2 or coh <= 2:
            dec = "reject"

        emit_decision(f"[{line_idx}] dec={dec} raw_dec={raw_dec} gnd={float(gnd):.2f} coh={float(coh):.2f} image={img_rel}")

        if dec == "keep":
            out = dict(item)
            out["judge"] = judge
            out["local_checks"] = {"attribute_aspects_touched": aspects, "has_causal_linking_terms": bool(has_linking)}
            return json.dumps(out, ensure_ascii=False), {"kept": 1}

        return None, {"drop_judge": 1}

    processed = 0

    # Process samples with thread pool.
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_sample, item): item for item in lines}

        for future in as_completed(futures):
            if pbar is not None:
                with pbar_lock:
                    pbar.update(1)

            try:
                output_line, counters = future.result()
                with lock:
                    processed += 1
                    kept += counters.get("kept", 0)
                    drop_hard += counters.get("drop_hard", 0)
                    drop_attr += counters.get("drop_attr", 0)
                    drop_judge += counters.get("drop_judge", 0)
                    judge_err += counters.get("judge_err", 0)
                    snapshot = (processed, kept, drop_hard, drop_attr, drop_judge, judge_err)

                if output_line:
                    output_queue.put(output_line)
            except Exception as e:
                print(f"Error processing sample: {e}", file=sys.stderr)
                with lock:
                    processed += 1
                    judge_err += 1
                    snapshot = (processed, kept, drop_hard, drop_attr, drop_judge, judge_err)

            if args.log_every > 0 and snapshot[0] % args.log_every == 0:
                if pbar is not None:
                    with pbar_lock:
                        pbar.set_postfix(
                            kept=snapshot[1],
                            hard_drop=snapshot[2],
                            attr_drop=snapshot[3],
                            judge_drop=snapshot[4],
                            judge_err=snapshot[5],
                        )
                else:
                    print(
                        f"[{snapshot[0]}] total={total} kept={snapshot[1]} hard_drop={snapshot[2]} "
                        f"attr_drop={snapshot[3]} judge_drop={snapshot[4]} judge_err={snapshot[5]}",
                        file=sys.stderr,
                    )

    if pbar is not None:
        pbar.close()

    with open(args.output, "w", encoding="utf-8") as fout:
        while not output_queue.empty():
            output_line = output_queue.get()
            fout.write(output_line + "\n")

    print(
        f"Done. total={total} kept={kept} hard_drop={drop_hard} attr_drop={drop_attr} judge_drop={drop_judge} judge_err={judge_err}",
        file=sys.stderr,
    )
    print(f"Output: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
