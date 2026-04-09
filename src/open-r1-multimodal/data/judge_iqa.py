#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iqa_judge.py (debuggable minimal)

Reads train_reversal_instruct.jsonl, applies hard-rule filtering, then calls
Qwen3-235B (OpenAI-compatible chat/completions) as a multimodal judge (English prompt).
Writes kept samples to train_reversal_instruct_judged.jsonl.

Key debug features:
- Drop reason counter summary at end
- Dump first N failing samples with helpful context (cot head / img_path)
- Optional per-sample realtime decision printing (--print_decisions)
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
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

TAG_THINK = re.compile(r"<think>(.*?)</think>", re.S | re.I)
TAG_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)
NUM_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d+)?|\.\d+)$")

# 1) Contamination rules: strict but avoid false positives
#    - block tool_call tags
#    - block code fences
#    - block <script ...>
#    - block role prefixes ONLY when they appear at line starts (avoid normal words in text)
ROLE_PREFIX_RE = re.compile(r"(^|\n)\s*(system|assistant|developer)\s*:", re.I)

# 2) Specific bad tags/markers (not "any <...>" to avoid false positives like "<3")
BAD_MARKERS_RE = re.compile(
    r"(<\s*tool_call\s*>|</\s*tool_call\s*>|<\s*system\s*>|</\s*system\s*>|<\s*assistant\s*>|</\s*assistant\s*>|```|<\s*script\b|\bBEGIN\b|\bEND\b)",
    re.I,
)

JUDGE_PROMPT_EN = r"""
You are an extremely strict and conservative IQA (Image Quality Assessment) data auditor.

Given an image, gt_score (ground-truth MOS), predicted_score (from <answer>), and cot_think (from <think>),
decide if this training sample is suitable.

Rules:
- Judge ONLY from the provided image. Do NOT invent details.
- Be conservative: any clear mismatch / technical misuse / inconsistency => reject.
- Output ONLY one strictly valid JSON object (no extra text, no markdown).

Score (0-5) with short evidence:
- visual_fidelity: are described artifacts (blur/noise/compression/banding/ringing/CA/exposure/color cast/etc.) truly supported?
- technical_rigor: are IQA terms used correctly (if used)? penalize concept confusion.
- score_consistency: does reasoning support predicted_score magnitude & direction on the given scale? consider gt_score too.

Output schema (STRICT):
{
  "decision": "keep" | "reject",
  "overall_confidence": 0-100,
  "scores": {"visual_fidelity":0-5,"technical_rigor":0-5,"score_consistency":0-5},
  "major_issues": ["..."],   // if reject, MUST be non-empty
  "minor_notes": ["..."],
  "debug": {"gt_score":<number>,"predicted_score":<number>,"abs_diff":<number>}
}

Guidance:
- If visual_fidelity <=2 OR score_consistency <=2 => strongly prefer reject.
- Hallucinated/contradicting defect claims => reject.
- Boilerplate with no verifiable cues => prefer reject.
""".strip()


def img_to_data_url(path: str) -> str:
    kind = imghdr.what(path) or os.path.splitext(path)[1].lower().lstrip(".")
    mime = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
    }.get(kind, f"image/{kind or 'jpeg'}")
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def hard_check(cot: str):
    """
    Returns: (think, ans_raw, pred_float, reason)
    """
    if not cot or not isinstance(cot, str):
        return None, None, None, "cot_empty_or_not_str"

    if BAD_MARKERS_RE.search(cot):
        return None, None, None, "contamination_bad_marker"

    if ROLE_PREFIX_RE.search(cot):
        return None, None, None, "contamination_role_prefix"

    t = TAG_THINK.findall(cot)
    a = TAG_ANSWER.findall(cot)
    if len(t) != 1 or len(a) != 1:
        return None, None, None, f"tag_count_invalid think={len(t)} answer={len(a)}"

    # exactly once each (open/close)
    if len(re.findall(r"<\s*think\s*>", cot, flags=re.I)) != 1 or len(re.findall(r"</\s*think\s*>", cot, flags=re.I)) != 1:
        return None, None, None, "think_tag_not_exactly_once"
    if len(re.findall(r"<\s*answer\s*>", cot, flags=re.I)) != 1 or len(re.findall(r"</\s*answer\s*>", cot, flags=re.I)) != 1:
        return None, None, None, "answer_tag_not_exactly_once"

    think = t[0].strip()
    ans_raw = a[0].strip()

    if not NUM_RE.match(ans_raw):
        return None, None, None, "answer_not_pure_number"

    try:
        pred = float(ans_raw)
    except Exception:
        return None, None, None, "answer_float_parse_error"

    # Light extra safety: think/answer shouldn't contain tool_call/system tags
    # (we already checked in BAD_MARKERS_RE, this is just redundancy)
    if BAD_MARKERS_RE.search(think) or BAD_MARKERS_RE.search(ans_raw):
        return None, None, None, "contamination_inside_fields"

    return think, ans_raw, pred, "ok"


def call_judge(api_url: str, api_key: str, model: str, image_data_url: str, user_text: str, timeout: int = 120) -> dict:
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
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ],
    }
    s = requests.Session()
    s.trust_env = False
    r = s.post(api_url, headers=headers, json=payload, timeout=timeout)
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"].strip()
    return json.loads(content)  # strict JSON-only expected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/chanxueyan/ano_people/datasets/Data-DeQA-Score/KONIQ/train_reversal_instruct.jsonl")
    ap.add_argument("--output", default="train_reversal_instruct_judged.jsonl")
    ap.add_argument("--image_root", default="/chanxueyan/ano_people/datasets/Data-DeQA-Score")
    ap.add_argument("--api_key", default="sk-lui3w1QQftJEKZ51596cA4E603A54398B5C0B44a666dE685")
    ap.add_argument("--api_url", default="http://192.168.9.180:8123/v1/chat/completions")
    ap.add_argument("--model", default="qwen3_VL_235B")
    ap.add_argument("--min_score", type=float, default=1.0)
    ap.add_argument("--max_score", type=float, default=100.0)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--max_lines", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--print_decisions", action="store_true", help="Print per-sample decision/drop reason in real time")
    ap.add_argument("--debug_dump_limit", type=int, default=30, help="Dump first N drop samples with context")
    ap.add_argument("--num_workers", type=int, default=10, help="Number of worker threads for parallel processing")
    args = ap.parse_args()

    if not args.api_key:
        print("ERROR: --api_key is empty (or set OPENAI_API_KEY env).", file=sys.stderr)
        sys.exit(1)

    kept = drop_hard = drop_judge = judge_err = total = 0
    reason_cnt = Counter()
    debug_dumps = 0
    
    # Thread-safe locks for shared resources
    lock = threading.Lock()
    pbar_lock = threading.Lock()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    # Read all lines first
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
    pbar = tqdm(total=progress_total, desc="Judging IQA", unit="sample", file=sys.stderr) if tqdm else None
    
    # Thread-safe output file
    output_queue = Queue()

    def emit(msg: str):
        if not args.print_decisions:
            return
        with pbar_lock:
            if pbar is not None:
                pbar.write(msg)
            else:
                print(msg, file=sys.stderr)

    def dump_fail(reason: str, img_rel=None, img_path=None, cot=None, extra=None):
        nonlocal debug_dumps
        with lock:
            if debug_dumps >= args.debug_dump_limit:
                return
            debug_dumps += 1
            cot_head = ""
            if isinstance(cot, str):
                cot_head = cot[:180].replace("\n", "\\n")
            parts = [f"[DUMP {debug_dumps}] reason={reason}"]
            if img_rel is not None:
                parts.append(f"img_rel={img_rel}")
            if img_path is not None:
                parts.append(f"img_path={img_path}")
            if cot_head:
                parts.append(f"cot_head={cot_head}")
            if extra:
                parts.append(f"extra={extra}")
            with pbar_lock:
                if pbar is not None:
                    pbar.write(" | ".join(parts))
                else:
                    print(" | ".join(parts), file=sys.stderr)

    def process_sample(item_data):
        """Process a single sample, returns (output_line, counters_update)"""
        line_idx, line = item_data
        
        try:
            item = json.loads(line)
        except Exception:
            with lock:
                reason_cnt["json_parse_error"] += 1
            emit(f"[{line_idx}] dec=drop_hard reason=json_parse_error")
            dump_fail("json_parse_error", cot=line)
            return None, {"drop_hard": 1}

        img_rel = item.get("image")
        gt = item.get("gt_score")
        cot = item.get("reversal_thinking") or item.get("cot_reasoning")

        if not isinstance(img_rel, str) or not isinstance(gt, (int, float)) or not isinstance(cot, str):
            with lock:
                reason_cnt["invalid_fields"] += 1
            emit(f"[{line_idx}] dec=drop_hard reason=invalid_fields")
            dump_fail("invalid_fields", img_rel=img_rel, cot=cot, extra=f"keys={list(item.keys())[:20]}")
            return None, {"drop_hard": 1}

        think, ans_raw, pred, reason = hard_check(cot)
        if reason != "ok":
            with lock:
                reason_cnt[f"hardcheck::{reason}"] += 1
            emit(f"[{line_idx}] dec=drop_hard reason={reason}")
            dump_fail(reason, img_rel=img_rel, cot=cot)
            return None, {"drop_hard": 1}

        if not (args.min_score <= float(gt) <= args.max_score and args.min_score <= float(pred) <= args.max_score):
            with lock:
                reason_cnt["score_out_of_range"] += 1
            emit(f"[{line_idx}] dec=drop_hard reason=score_out_of_range gt={gt} pred={pred}")
            dump_fail("score_out_of_range", img_rel=img_rel, cot=cot, extra=f"gt={gt},pred={pred},range=[{args.min_score},{args.max_score}]")
            return None, {"drop_hard": 1}

        img_path = img_rel if os.path.isabs(img_rel) else os.path.join(args.image_root, img_rel)
        if not os.path.exists(img_path):
            with lock:
                reason_cnt["image_not_found"] += 1
            emit(f"[{line_idx}] dec=drop_hard reason=image_not_found image={img_rel}")
            dump_fail("image_not_found", img_rel=img_rel, img_path=img_path, cot=cot)
            return None, {"drop_hard": 1}

        # Call LLM judge
        try:
            image_data_url = img_to_data_url(img_path)
            user_text = (
                f"scale_min: {args.min_score}\nscale_max: {args.max_score}\n"
                f"gt_score: {float(gt)}\npredicted_score: {float(pred)}\n"
                f"cot_think:\n{think}\n"
            )
            judge = call_judge(args.api_url, args.api_key, args.model, image_data_url, user_text, args.timeout)
        except Exception as e:
            with lock:
                reason_cnt["judge_err_api_or_json"] += 1
            emit(f"[{line_idx}] dec=judge_err reason=api_or_parse_exception image={img_rel}")
            dump_fail("judge_err_api_or_json", img_rel=img_rel, img_path=img_path, extra=str(e)[:300])
            return None, {"judge_err": 1}

        # Validate judge output
        if not isinstance(judge, dict):
            with lock:
                reason_cnt["judge_not_dict"] += 1
            emit(f"[{line_idx}] dec=drop_judge reason=judge_not_dict image={img_rel}")
            dump_fail("judge_not_dict", img_rel=img_rel, extra=str(judge)[:200])
            return None, {"drop_judge": 1}

        dec = judge.get("decision")
        scores = judge.get("scores")
        if not isinstance(scores, dict):
            with lock:
                reason_cnt["judge_scores_not_dict"] += 1
            emit(f"[{line_idx}] dec=drop_judge reason=scores_not_dict image={img_rel}")
            dump_fail("judge_scores_not_dict", img_rel=img_rel, extra=str(judge)[:300])
            return None, {"drop_judge": 1}

        vf = scores.get("visual_fidelity")
        sc = scores.get("score_consistency")
        if dec not in ("keep", "reject") or not isinstance(vf, (int, float)) or not isinstance(sc, (int, float)):
            with lock:
                reason_cnt["judge_schema_invalid"] += 1
            emit(f"[{line_idx}] dec=drop_judge reason=judge_schema_invalid image={img_rel}")
            dump_fail("judge_schema_invalid", img_rel=img_rel, extra=str(judge)[:400])
            return None, {"drop_judge": 1}

        raw_dec = dec
        if vf <= 2 or sc <= 2:
            dec = "reject"

        emit(f"[{line_idx}] dec={dec} raw_dec={raw_dec} vf={float(vf):.2f} sc={float(sc):.2f} image={img_rel}")

        if dec == "keep":
            out = dict(item)
            out["judge"] = judge
            return json.dumps(out, ensure_ascii=False), {"kept": 1}
        else:
            with lock:
                reason_cnt["judge_reject"] += 1
            return None, {"drop_judge": 1}
    
    
    
    # Process samples with thread pool
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_sample, item): item for item in lines}
        
        for future in as_completed(futures):
            if pbar is not None:
                with pbar_lock:
                    pbar.update(1)

            try:
                output_line, counters = future.result()
                if output_line:
                    output_queue.put(output_line)
                    with lock:
                        kept += counters.get("kept", 0)
                else:
                    with lock:
                        drop_hard += counters.get("drop_hard", 0)
                        drop_judge += counters.get("drop_judge", 0)
                        judge_err += counters.get("judge_err", 0)
            except Exception as e:
                print(f"Error processing sample: {e}", file=sys.stderr)
                with lock:
                    judge_err += 1
    
    if pbar is not None:
        pbar.close()
    
    # Write all outputs to file
    with open(args.output, "w", encoding="utf-8") as fout:
        while not output_queue.empty():
            output_line = output_queue.get()
            fout.write(output_line + "\n")

    print(f"Done. total={total} kept={kept} hard_drop={drop_hard} judge_drop={drop_judge} judge_err={judge_err}", file=sys.stderr)
    print(f"Output: {args.output}", file=sys.stderr)

    print("\nDrop reason summary:", file=sys.stderr)
    for k, v in reason_cnt.most_common():
        print(f"  {k}: {v}", file=sys.stderr)


if __name__ == "__main__":
    main()