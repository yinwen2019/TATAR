#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_uni_iqa_iaa.py  (FINAL)

Fixes the real root cause you described: uneven per-rank compute cost -> rank0 much slower
-> other ranks hit NCCL barrier and timeout retrieving ncclUniqueId from rank0.

What this final version changes (without changing your desired SFT output with <think>):
1) Cost-balanced sharding (LPT greedy) based on per-image file size (proxy for pixels/token cost).
   This makes each rank's total expected workload much closer.
2) Remove NCCL barrier entirely (the 600s store timeout trap).
   Instead use file-based rendezvous: each rank writes a .done file after finishing.
   Rank0 waits for all done files then merges & scores.
3) Stop spamming stdout per sample (print(comp) is extremely slow and amplifies rank skew).
   Provide --debug_print to print a few samples only on rank0.
4) Add optional --wait_timeout / --poll_interval for rank0 rendezvous.

NOTE: This is an eval script; file-based sync is robust and does not depend on NCCL comm setup.
"""

import os
import re
import json
import time
import argparse
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
)
from trl.data_utils import maybe_apply_chat_template
from tqdm import tqdm


# =============================================================================
# Hotfix: match training (FlashAttention bug workaround)
# =============================================================================
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (  # noqa: E402
    Qwen2_5_VLVisionFlashAttention2,
    apply_rotary_pos_emb_flashatt,
    flash_attn_varlen_func,
)
from typing import Tuple as _Tuple  # noqa: E402


def _qwen25vl_custom_forward(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    position_embeddings: Optional[_Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    seq_length = hidden_states.shape[0]
    q, k, v = (
        self.qkv(hidden_states)
        .reshape(seq_length, 3, self.num_heads, -1)
        .permute(1, 0, 2, 3)
        .unbind(0)
    )
    if position_embeddings is None:
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        cos = emb.cos().float()
        sin = emb.sin().float()
    else:
        cos, sin = position_embeddings
        cos = cos.to(torch.float)
        sin = sin.to(torch.float)

    q, k = apply_rotary_pos_emb_flashatt(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
    q = q.squeeze(0)
    k = k.squeeze(0)

    max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
    attn_output = flash_attn_varlen_func(
        q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen
    ).reshape(seq_length, -1)
    attn_output = self.proj(attn_output)
    return attn_output


Qwen2_5_VLVisionFlashAttention2.forward = _qwen25vl_custom_forward


# =============================================================================
# Datasets
# =============================================================================
DATASETS = {
    # IQA datasets
    "KonIQ": dict(
        name="KonIQ",
        json_path="benchmark/VR/IQA/KonIQ-10K/KonIQ-10K.json",
        img_root="/chanxueyan/ano_people/datasets/Data-DeQA-Score/KONIQ/images",
        task="iqa_score",
        img_key="image",
        gt_key="gt_score",
    ),
    "KADID": dict(
        name="KADID",
        json_path="benchmark/VR/IQA/KADID/KADID.json",
        img_root="/chanxueyan/ano_people/datasets/Data-DeQA-Score/kadid10k/images",
        task="iqa_score",
        img_key="image",
        gt_key="gt_score",
    ),
    "PIPAL": dict(
        name="PIPAL",
        json_path="benchmark/VR/IQA/PIPAL/PIPAL.json",
        img_root="",
        task="iqa_score",
        img_key="image",
        gt_key="gt_score",
    ),
    "SPAQ": dict(
        name="SPAQ",
        json_path="benchmark/VR/IQA/SPAQ/SPAQ.json",
        img_root="/chanxueyan/ano_people/datasets/Data-DeQA-Score/SPAQ/TestImage",
        task="iqa_score",
        img_key="image",
        gt_key="gt_score",
    ),
    # IAA datasets
    "ArtiMuse": dict(
        name="ArtiMuse",
        json_path="benchmark/VR/IAA/ArtiMuse-10K/ArtiMuse-10K.json",
        img_root="/chanxueyan/ano_people/datasets/ArtiMuse-10K-251219/images",
        task="iaa_score",
        img_key="image",
        gt_key="gt_score",
    ),
    "AVA": dict(
        name="AVA",
        json_path="benchmark/VR/IAA/AVA/AVA.json",
        img_root="/chanxueyan/ano_people/datasets/IAA/AVA_dataset/image",
        task="iaa_score",
        img_key="image",
        gt_key="gt_score",
    ),
    "FLICKR-AES": dict(
        name="FLICKR-AES",
        json_path="benchmark/VR/IAA/FLICKR-AES/FLICKR-AES.json",
        img_root="/chanxueyan/ano_people/datasets/IAA/40K",
        task="iaa_score",
        img_key="image",
        gt_key="gt_score",
    ),
    "TAD66K": dict(
        name="TAD66K",
        json_path="benchmark/VR/IAA/TAD66K/TAD66K.json",
        img_root="/chanxueyan/ano_people/datasets/IAA/TAD66K",
        task="iaa_score",
        img_key="image",
        gt_key="gt_score",
    ),
}

# =============================================================================
# PROMPTS (same as training)
# =============================================================================
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it.\n"
    "The Assistant may include optional reasoning inside <think>...</think> (it may be brief or empty).\n"
    "The final answer MUST be provided inside <answer>...</answer>.\n"
    "Inside <answer>, output ONLY a single numeric rating (a float), with no extra text.\n"
    "If <think> is present, it must appear before <answer>."
)

IQA_QUESTION_PROMPT = (
    "TASK: Image Quality Assessment (IQA).\n"
    "What is your overall rating on the quality of the given picture?\n"
    "The rating should be a float between 0 and 100, rounded to two decimal places, "
    "with 0 representing very poor quality and 100 representing excellent quality.\n"
    "You may answer directly without detailed reasoning.\n"
    "In <answer>, output ONLY the numeric score (e.g., 73.42)."
)

IAA_QUESTION_PROMPT = (
    "TASK: Image Aesthetic Assessment (IAA).\n"
    "What is your overall rating on the aesthetic quality of the given picture?\n"
    "The rating should be a float between 0 and 100, rounded to two decimal places, "
    "with 0 representing very poor aesthetics and 100 representing excellent aesthetics.\n"
    "Consider composition, lighting, color harmony, subject presentation, balance, style, and emotional impact.\n"
    "In <answer>, output ONLY the numeric score (e.g., 73.42)."
)


def build_messages(task: str) -> List[Dict[str, Any]]:
    prompt_text = IQA_QUESTION_PROMPT if task == "iqa_score" else IAA_QUESTION_PROMPT
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]},
    ]


# =============================================================================
# GT / Pred mapping (same as training)
# =============================================================================
def gt_to_100(task: str, gt):
    if gt is None:
        return None
    try:
        x = float(gt)
    except Exception:
        return None

    if task == "iqa_score":
        if 0.0 <= x <= 1.0:
            return x * 100.0
        if 1.0 <= x <= 5.0:
            return (x - 1.0) / 4.0 * 100.0
        return max(0.0, min(100.0, x))

    if task == "iaa_score":
        if 0.0 <= x <= 10.0:
            return x * 10.0
        return max(0.0, min(100.0, x))

    return None


def pred_to_100(task: str, pred: float) -> float:
    x = float(pred)
    if 0.0 <= x <= 1.0:
        return x * 100.0
    if task == "iqa_score" and 1.0 <= x <= 5.0:
        return (x - 1.0) / 4.0 * 100.0
    if task == "iaa_score" and 0.0 <= x <= 10.0:
        return x * 10.0
    return max(0.0, min(100.0, x))


# =============================================================================
# Parse ONLY from <answer>
# =============================================================================
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def parse_pred_from_completion(text: str) -> Optional[float]:
    if not isinstance(text, str):
        return None

    matches = ANSWER_RE.findall(text)
    if not matches:
        return None

    ans = matches[-1].strip()
    ans = re.sub(r"^```(?:json)?\s*", "", ans, flags=re.IGNORECASE)
    ans = re.sub(r"\s*```$", "", ans)

    # tolerate JSON {"rating": ...}
    try:
        obj = json.loads(ans)
        if isinstance(obj, dict) and "rating" in obj:
            return float(obj["rating"])
    except Exception:
        pass

    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", ans)
    if m:
        return float(m.group(0))
    return None


# =============================================================================
# Metrics
# =============================================================================
def pearsonr(x, y):
    x = x - x.mean()
    y = y - y.mean()
    denom = (np.sqrt((x * x).sum()) * np.sqrt((y * y).sum()))
    if denom == 0:
        return float("nan")
    return float((x * y).sum() / denom)


def spearmanr(x, y):
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return pearsonr(rx.astype(np.float64), ry.astype(np.float64))


# =============================================================================
# Dataset / Cost-balanced Sampler (LPT)
# =============================================================================
class UniEvalDataset(Dataset):
    """
    Loads JSON and provides image + gt_100.
    Builds valid indices by (path exists, gt exists).
    Also exposes a per-sample "cost" proxy for balancing (file size in bytes).
    """

    def __init__(self, json_path, img_root, task, img_key, gt_key):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.data = data
        self.img_root = img_root
        self.task = task
        self.img_key = img_key
        self.gt_key = gt_key

        self.valid_indices = self._build_valid_indices()
        self._costs = self._build_costs()  # aligned with valid_indices

    def _resolve_path(self, rel: str) -> str:
        if os.path.isabs(rel):
            return rel
        return os.path.join(self.img_root, rel)

    def _build_valid_indices(self) -> List[int]:
        valid = []
        for i, ex in enumerate(self.data):
            rel = ex.get(self.img_key)
            gt = gt_to_100(self.task, ex.get(self.gt_key))
            if rel is None or gt is None:
                continue
            img_path = self._resolve_path(rel)
            if not os.path.exists(img_path):
                continue
            valid.append(i)
        return valid

    def _build_costs(self) -> List[int]:
        costs = []
        for idx in self.valid_indices:
            rel = self.data[idx].get(self.img_key)
            img_path = self._resolve_path(rel) if rel is not None else ""
            try:
                costs.append(int(os.path.getsize(img_path)))
            except Exception:
                costs.append(0)
        return costs

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, j):
        idx = self.valid_indices[j]
        ex = self.data[idx]

        rel = ex.get(self.img_key)
        gt = gt_to_100(self.task, ex.get(self.gt_key))

        if rel is None or gt is None:
            return {"bad": True}

        img_path = self._resolve_path(rel)
        if not os.path.exists(img_path):
            return {"bad": True}

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            return {"bad": True}

        return {
            "bad": False,
            "idx": idx,  # original index in json (stable merge key)
            "image": img,
            "image_path": img_path,
            "gt_100": gt,
        }

    @property
    def costs(self) -> List[int]:
        return self._costs


class LPTBalancedSampler(Sampler[int]):
    """
    Longest Processing Time (LPT) greedy bin-packing.
    Distributes samples across ranks to balance total cost.

    Works on dataset indices j in [0, len(dataset)), where cost[j] is dataset.costs[j].
    """

    def __init__(self, costs: List[int], rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.n = len(costs)
        self._assignments = self._lpt_assign(costs, world_size)
        self._my = self._assignments[rank]

    @staticmethod
    def _lpt_assign(costs: List[int], world_size: int) -> List[List[int]]:
        # items: (cost, j)
        items = [(int(c), j) for j, c in enumerate(costs)]
        items.sort(key=lambda x: x[0], reverse=True)

        sums = [0] * world_size
        bins: List[List[int]] = [[] for _ in range(world_size)]

        for c, j in items:
            k = int(np.argmin(sums))
            bins[k].append(j)
            sums[k] += c

        return bins

    def __iter__(self):
        return iter(self._my)

    def __len__(self):
        return len(self._my)


# =============================================================================
# Run one rank
# =============================================================================
@torch.no_grad()
def run_rank(args, cfg) -> str:
    device = torch.device("cuda", args.local_rank)

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, use_fast=False
    )

    # Align padding behavior (important for left-padding generation slicing)
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.padding_side = "left"
    tokenizer.padding_side = "left"

    # MUST match training
    processor.image_processor.max_pixels = args.max_pixels
    processor.image_processor.min_pixels = args.min_pixels

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
    ).eval().to(device)

    dataset = UniEvalDataset(
        cfg["json_path"],
        cfg["img_root"],
        cfg["task"],
        cfg["img_key"],
        cfg["gt_key"],
    )

    # Balanced sampler to reduce rank skew (the real fix)
    sampler = LPTBalancedSampler(dataset.costs, args.rank, args.world_size)

    # Build messages and precompute prompt ONCE (same prompt for all samples)
    messages = build_messages(cfg["task"])
    prompt_text = maybe_apply_chat_template({"prompt": messages}, processor)["prompt"]

    def collate(batch):
        batch = [b for b in batch if not b.get("bad", False)]
        if not batch:
            return None

        prompts_text = [prompt_text for _ in batch]  # batch_size times
        images = [b["image"] for b in batch]
        inputs = processor(
            text=prompts_text,
            images=images,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        return inputs, batch

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=collate,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False,
        persistent_workers=(args.num_workers > 0 and args.persistent_workers),
    )

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{cfg['name']}.rank{args.rank}.jsonl")
    done_path = os.path.join(args.out_dir, f"{cfg['name']}.rank{args.rank}.done")

    # For slicing model outputs: compute prompt_len per batch
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id

    debug_left = int(args.debug_print)

    with open(out_path, "w", encoding="utf-8") as f:
        pbar = tqdm(loader, desc=f"{cfg['name']} rank{args.rank}", ncols=110)
        for pack in pbar:
            if pack is None:
                continue

            inputs, batch_items = pack
            inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

            prompt_len = inputs["input_ids"].shape[1]
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                eos_token_id=eos_id,
                pad_token_id=pad_id,
            )

            # decode only newly generated tokens
            for o, b in zip(outputs, batch_items):
                comp = tokenizer.decode(o[prompt_len:], skip_special_tokens=True)
                pred = parse_pred_from_completion(comp)

                pred100 = None
                if pred is not None and np.isfinite(pred):
                    pred100 = float(pred_to_100(cfg["task"], pred))

                if args.rank == 0 and debug_left > 0:
                    tqdm.write("---- completion (rank0 debug) ----")
                    tqdm.write(comp)
                    tqdm.write(f"pred_100: {pred100} | gt_100: {b['gt_100']}")
                    tqdm.write("-------------------------------")
                    debug_left -= 1

                f.write(
                    json.dumps(
                        {
                            "idx": b["idx"],
                            "image_path": b["image_path"],
                            "gt_100": b["gt_100"],
                            "pred_100": pred100,
                            "comp":comp,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    # Write a done marker (atomic best-effort)
    try:
        with open(done_path, "w", encoding="utf-8") as df:
            df.write("ok\n")
    except Exception:
        pass

    return out_path


def wait_all_done(args, cfg) -> bool:
    """Rank0 waits for all ranks to finish via .done files."""
    expected = [os.path.join(args.out_dir, f"{cfg['name']}.rank{r}.done") for r in range(args.world_size)]
    t0 = time.time()
    while True:
        missing = [p for p in expected if not os.path.exists(p)]
        if not missing:
            return True
        if time.time() - t0 > args.wait_timeout:
            print("[ERROR] timeout waiting .done files. Missing:")
            for p in missing:
                print("  ", p)
            return False
        time.sleep(args.poll_interval)


def merge_and_score(args, cfg):
    recs = {}
    missing = []
    for r in range(args.world_size):
        p = os.path.join(args.out_dir, f"{cfg['name']}.rank{r}.jsonl")
        if not os.path.exists(p):
            missing.append(p)
            continue
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                recs[obj["idx"]] = obj

    if missing:
        print("[WARN] missing rank outputs:")
        for p in missing:
            print("  ", p)

    y_true, y_pred = [], []
    for k in sorted(recs):
        if recs[k].get("pred_100", None) is not None:
            y_true.append(recs[k]["gt_100"])
            y_pred.append(recs[k]["pred_100"])

    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)

    metrics = {
        "dataset": cfg["name"],
        "valid": int(len(y_true)),
        "SRCC": spearmanr(y_true, y_pred) if len(y_true) >= 3 else float("nan"),
        "PLCC": pearsonr(y_true, y_pred) if len(y_true) >= 3 else float("nan"),
    }

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    with open(os.path.join(args.out_dir, f"{cfg['name']}.metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def get_dist_info():
    if "WORLD_SIZE" in os.environ and "RANK" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return rank, local_rank, world_size
    return 0, 0, 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset", required=True, choices=sorted(list(DATASETS.keys())))
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--out_dir", default="results/qwen_vr_eval")

    # align training pixels
    parser.add_argument("--max_pixels", type=int, default=1572864)
    parser.add_argument("--min_pixels", type=int, default=3136)

    # dataloader
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true", help="Enable pin_memory (default off for stability)")
    parser.add_argument("--persistent_workers", action="store_true", help="Enable persistent_workers (default off)")

    # debug: print a few samples on rank0 (avoid massive stdout)
    parser.add_argument("--debug_print", type=int, default=2, help="Print N completions on rank0 for debugging")

    # rendezvous (rank0 wait all .done)
    parser.add_argument("--wait_timeout", type=int, default=6 * 3600, help="Seconds rank0 waits for all ranks")
    parser.add_argument("--poll_interval", type=float, default=5.0, help="Seconds between polling done files")

    # optional override
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--world_size", type=int, default=None)
    parser.add_argument("--local_rank", type=int, default=None)

    args = parser.parse_args()

    env_rank, env_local_rank, env_world_size = get_dist_info()
    args.rank = env_rank if args.rank is None else int(args.rank)
    args.world_size = env_world_size if args.world_size is None else int(args.world_size)
    args.local_rank = env_local_rank if args.local_rank is None else int(args.local_rank)

    if torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)

    # init dist for rank/world info only; we won't use barrier (avoids NCCL store timeout)
    if args.world_size > 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend="nccl",
            rank=args.rank,
            world_size=args.world_size,
        )

    cfg = DATASETS[args.dataset]

    # run eval per rank
    run_rank(args, cfg)

    # file-based rendezvous + merge only on rank0
    if args.rank == 0:
        ok = wait_all_done(args, cfg)
        if ok:
            merge_and_score(args, cfg)

    if args.world_size > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()