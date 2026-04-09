#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os

# 你数据里 human prompt 的属性名（精确匹配）
ATTR_MAP = {
    "Composition & Structure": "composition_structure",
    "Visual Elements": "visual_elements",
    "Technical Execution": "technical_execution",
    "Originality & Creativity": "originality_creativity",
    "Theme & Communication": "theme_communication",
    "Emotion & Viewer Response": "emotion_viewer_response",
    "Overall Gestalt": "overall_gestalt",
    "Overall Evaluation": "overall_evaluation",
}

ASPECT_PREFIX = "Please evaluate the aesthetic quality of this image from the aespect of "

def extract_attr_name_from_prompt(prompt: str) -> str:
    """
    你的数据格式统一：prompt 中包含 `from the aspect of {ATTR}`
    这里直接截取 {ATTR} 到行尾或句末标点。
    """
    idx = prompt.find(ASPECT_PREFIX)
    if idx < 0:
        raise ValueError(f"Cannot find prefix `{ASPECT_PREFIX}` in prompt: {prompt[:160]}")

    s = prompt[idx + len(ASPECT_PREFIX):].strip()

    # 根据你的统一格式，通常属性名在这一行结束；这里做最轻量的截断
    for sep in ["\n", "\r", ".", "?", "。", "？"]:
        j = s.find(sep)
        if j >= 0:
            s = s[:j].strip()
    return s

def load_text_jsonl(text_path: str) -> dict:
    """
    Returns:
      img2attrs: {image_basename: {canonical_key: rationale}}
    """
    img2attrs = {}

    with open(text_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            img_base = os.path.basename(obj["image"])

            conv = obj["conversations"]
            human = conv[0]["value"]
            gpt = conv[1]["value"].strip()

            attr_name = extract_attr_name_from_prompt(human)

            if attr_name not in ATTR_MAP:
                raise KeyError(
                    f"[Line {line_no}] Unknown attr name: `{attr_name}`. "
                    f"Expected one of: {list(ATTR_MAP.keys())}"
                )

            key = ATTR_MAP[attr_name]

            if img_base not in img2attrs:
                img2attrs[img_base] = {}

            # 若同一 image+attr 多次出现：后者覆盖前者（也可改成保留更长的一条）
            img2attrs[img_base][key] = gpt

    return img2attrs

def merge(score_json_path: str, img2attrs: dict, out_path: str):
    with open(score_json_path, "r", encoding="utf-8") as f:
        score_data = json.load(f)

    hit = 0
    for item in score_data:
        img_base = os.path.basename(item["image"])
        attrs = img2attrs.get(img_base)
        if attrs:
            item["aesthetic_attributes"] = attrs
            hit += 1

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(score_data, f, ensure_ascii=False, indent=2)

    print(f"Done. merged {hit}/{len(score_data)} items with attributes.")
    print(f"Output: {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text_jsonl", required=True, help="Path to text/train.jsonl")
    ap.add_argument("--score_json", required=True, help="Path to score/train.json")
    ap.add_argument("--out", default=None, help="Output path (default: train.with_attrs.json in score dir)")
    args = ap.parse_args()

    if args.out is None:
        out_dir = os.path.dirname(os.path.abspath(args.score_json))
        args.out = os.path.join(out_dir, "train.with_attrs.json")

    img2attrs = load_text_jsonl(args.text_jsonl)
    merge(args.score_json, img2attrs, args.out)

if __name__ == "__main__":
    main()
