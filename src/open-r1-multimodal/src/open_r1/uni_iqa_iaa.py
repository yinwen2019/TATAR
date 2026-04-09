# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset

from open_r1.trainer import Qwen2VLGRPOTrainerUni, GRPOConfig
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
import yaml
import json
import random
import math
import torch

# ----------------------- Fix the flash attention bug in the current version of transformers -----------------------
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLVisionFlashAttention2,
    apply_rotary_pos_emb_flashatt,
    flash_attn_varlen_func,
)

def custom_forward(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    seq_length = hidden_states.shape[0]
    q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    if position_embeddings is None:
        # logger.warning_once(...)  # left as-is in upstream; logger may be undefined here
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
    attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
        seq_length, -1
    )
    attn_output = self.proj(attn_output)
    return attn_output

Qwen2_5_VLVisionFlashAttention2.forward = custom_forward


# ----------------------- Main Script -----------------------
@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """
    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "rank", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    image_root: Optional[str] = field(
        default=None,
        metadata={"help": "Root directory of the image"},
    )
    score_reward_threshold: Optional[float] = field(
        default=10.0,
        metadata={"help": "Threshold for score reward on 0-100 scale"},
    )
    dataset_iaa: Optional[str] = field(
        default=None,
        metadata={"help": "YAML file path for the IAA scoring dataset"},
    )
    dataset_iqa: Optional[str] = field(
        default=None,
        metadata={"help": "YAML file path for the IQA scoring dataset"},
    )

# Make script_args available to reward functions (trainer calls them without passing script_args)
_GLOBAL_SCRIPT_ARGS: Optional[GRPOScriptArguments] = None


# ----------------------- Prompts (UPDATED: <answer> outputs a single number) -----------------------
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it.\n"
    "The Assistant may include optional reasoning inside <think>...</think> (it may be brief or detailed).\n"
    "The final answer MUST be provided inside <answer>...</answer>.\n"
    "Inside <answer>, output ONLY a single numeric rating (a float), with no extra text.\n"
    "If <think> is present, it must appear before <answer>."
)

IQA_QUESTION_PROMPT = (
    "TASK: Image Quality Assessment (IQA).\n"
    "What is your overall rating on the quality of the given picture?\n"
    "The rating should be a float between 0 and 100, rounded to two decimal places, "
    "with 0 representing very poor quality and 100 representing excellent quality.\n"
    "You may answer briefly without detailed reasoning.\n"
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


def gt_to_100(task: str, gt):
    """
    Map ground-truth scores to a unified 0-100 scale.
    - IQA: gt_score_norm might be in [0,1] or [1,5] (or already 0-100).
    - IAA: gt_score usually in [1,10] (or already 0-100).
    """
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
    elif task == "iaa_score":
        if 0.0 <= x <= 10.0:
            return x * 10.0
        return max(0.0, min(100.0, x))
    else:
        return max(0.0, min(100.0, x))


# ----------------------- Parsing (UPDATED: parse number directly) -----------------------
_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")

def parse_rating_from_completion(content: str) -> Optional[float]:
    """
    Extract rating from <answer>...</answer> as a single number.
    - Tolerates whitespace/newlines.
    - Rejects if <answer> contains multiple numbers or non-numeric junk (beyond whitespace).
    Returns float or None.
    """
    if not isinstance(content, str):
        return None

    m = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL | re.IGNORECASE)
    if not m:
        return None

    ans = m.group(1).strip()

    # Strip common code fences if present (some models still do this)
    ans = re.sub(r"^```(?:text)?\s*", "", ans, flags=re.IGNORECASE).strip()
    ans = re.sub(r"\s*```$", "", ans).strip()

    # Must be (essentially) a single number
    nums = _NUM_RE.findall(ans)
    if len(nums) != 1:
        return None

    # Ensure the answer string doesn't contain other non-space characters besides the number
    # (e.g., "73.42 points" should be rejected)
    ans_clean = _NUM_RE.sub("", ans).strip()
    if ans_clean != "":
        return None

    try:
        return float(nums[0])
    except Exception:
        return None


class LazySupervisedDataset(Dataset):
    def __init__(self, script_args: GRPOScriptArguments):
        super(LazySupervisedDataset, self).__init__()
        self.script_args = script_args

        self.iaa_samples = []
        self.iqa_samples = []
        if script_args.dataset_iaa:
            self.iaa_samples = self._load_samples_from_yaml(script_args.dataset_iaa)
        if script_args.dataset_iqa:
            self.iqa_samples = self._load_samples_from_yaml(script_args.dataset_iqa)

        if not self.iaa_samples and not self.iqa_samples:
            raise ValueError("At least one dataset file must be provided: --dataset_iaa or --dataset_iqa")

        self.total_len = len(self.iaa_samples) + len(self.iqa_samples)

    def _load_samples_from_yaml(self, data_path: str):
        samples = []
        if not data_path.endswith(".yaml"):
            raise ValueError(f"Unsupported file type: {data_path}")
        with open(data_path, "r") as file:
            yaml_data = yaml.safe_load(file)
            datasets = yaml_data.get("datasets", [])
            for data in datasets:
                json_path = data.get("json_path")
                sampling_strategy = data.get("sampling_strategy", "all")
                sampling_number = None

                if json_path.endswith(".jsonl"):
                    cur_data_dict = []
                    with open(json_path, "r") as json_file:
                        for line in json_file:
                            cur_data_dict.append(json.loads(line.strip()))
                elif json_path.endswith(".json"):
                    with open(json_path, "r") as json_file:
                        cur_data_dict = json.load(json_file)
                else:
                    raise ValueError(f"Unsupported file type: {json_path}")

                if ":" in sampling_strategy:
                    sampling_strategy, sampling_number = sampling_strategy.split(":")
                    if "%" in sampling_number:
                        sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
                    else:
                        sampling_number = int(sampling_number)

                if sampling_strategy == "first" and sampling_number is not None:
                    cur_data_dict = cur_data_dict[:sampling_number]
                elif sampling_strategy == "end" and sampling_number is not None:
                    cur_data_dict = cur_data_dict[-sampling_number:]
                elif sampling_strategy == "random" and sampling_number is not None:
                    random.shuffle(cur_data_dict)
                    cur_data_dict = cur_data_dict[:sampling_number]

                print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
                samples.extend(cur_data_dict)
        return samples

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        if self.iqa_samples and self.iaa_samples:
            if index < len(self.iqa_samples):
                chosen_task = "iqa_score"
                example = self.iqa_samples[index]
                solution = example.get("gt_score_norm", None)
                prompt_text = IQA_QUESTION_PROMPT
            else:
                chosen_task = "iaa_score"
                index2 = index - len(self.iqa_samples)
                if index2 >= len(self.iaa_samples):
                    raise IndexError("Index out of range for iaa_samples")
                example = self.iaa_samples[index2]
                solution = example.get("gt_score", None)
                prompt_text = IAA_QUESTION_PROMPT
        elif self.iqa_samples:
            chosen_task = "iqa_score"
            example = self.iqa_samples[index]
            solution = example.get("gt_score_norm", None)
            prompt_text = IQA_QUESTION_PROMPT
        elif self.iaa_samples:
            chosen_task = "iaa_score"
            example = self.iaa_samples[index]
            solution = example.get("gt_score", None)
            prompt_text = IAA_QUESTION_PROMPT
        else:
            raise ValueError("No available dataset (iqa_samples or iaa_samples)")

        solution_100 = gt_to_100(chosen_task, solution)
        sample = {"task": chosen_task, "solution": solution_100}

        image = None
        image_root = self.script_args.image_root
        max_retry = 20

        def build_path(ex, task):
            rel = ex["image"]
            if os.path.isabs(rel):
                return rel
            if task == "iqa_score":
                return os.path.join(image_root, "Data-DeQA-Score", rel)
            else:
                return os.path.join(image_root, "ArtiMuse-10K-251219/images", rel)

        for _ in range(max_retry):
            try:
                image_path = build_path(example, chosen_task)
                with Image.open(image_path) as im:
                    im = im.convert("RGB")
                    image = im.copy()
                break
            except Exception:
                if chosen_task == "iqa_score":
                    example = self.iqa_samples[random.randint(0, len(self.iqa_samples) - 1)]
                    solution_100 = gt_to_100("iqa_score", example.get("gt_score_norm", None))
                else:
                    example = self.iaa_samples[random.randint(0, len(self.iaa_samples) - 1)]
                    solution_100 = gt_to_100("iaa_score", example.get("gt_score", None))

        if image is None:
            raise RuntimeError(f"Failed to load image after {max_retry} retries. Last path: {image_path}")

        sample["image"] = image
        sample["image_path"] = image_path
        sample["solution"] = solution_100

        sample["prompt"] = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]},
        ]
        return sample


# ----------------------- Rewards (UPDATED: numeric answer) -----------------------
def score_reward_old(completions, solution, task, image_path, **kwargs):
    """
    Unified reward for IQA & IAA on 0-100 scale:
      - Parse pred_100 from <answer> as a single numeric score
      - Reward = 1.0 if abs(pred_100 - gt_100) < threshold (default 10.0)
    """
    contents = [completion[0]["content"] for completion in completions]
    rewards = []

    # The trainer may expand task/solution per generation; keep your original subsampling logic
    num_gen = len(task) // len(contents)
    subsampled_tasks = task[::num_gen]
    subsampled_solutions = solution[::num_gen]
    subsampled_image_paths = image_path[::num_gen]

    threshold = float((_GLOBAL_SCRIPT_ARGS.score_reward_threshold if _GLOBAL_SCRIPT_ARGS else 10.0))

    for i, (t, content, true_sol) in enumerate(zip(subsampled_tasks, contents, subsampled_solutions)):
        reward = 0.0
        pred = None
        try:
            pred = parse_rating_from_completion(content)
            if pred is not None and true_sol is not None:
                pred = max(0.0, min(100.0, float(pred)))
                gt = max(0.0, min(100.0, float(true_sol)))
                if abs(pred - gt) < threshold:
                    reward = 1.0
        except Exception as e:
            print("Error in computing reward", e)

        rewards.append(reward)

        if os.getenv("DEBUG_MODE") == "true":
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                current_rank = torch.distributed.get_rank()
            else:
                current_rank = 0
            current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"------------- {current_time} Rank: {current_rank} Task: {t} Reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Image Path: {subsampled_image_paths[i]}\n")
                if pred is not None:
                    f.write(f"Model Score(0-100): {pred}\n")
                f.write(f"Ground Truth(0-100): {true_sol}\n")

    return rewards


# ----------------------- Rewards (Gaussian soft accuracy + hard length constraints) -----------------------
def score_reward(completions, solution, task, image_path, **kwargs):
    contents = [c[0]["content"] for c in completions]
    rewards = []

    num_gen = len(task) // len(contents)
    tasks = task[::num_gen]
    gts   = solution[::num_gen]

    # threshold：你原来的误差阈值（5/10之类）
    # target：当误差=threshold 时，希望reward大概是多少（0.5最常用）
    # alpha：自适应强度，越大越“宽容”（误差越大，sigma_dyn越大，reward下降越慢）
    target = 0.5
    alpha_iqa = 0.8
    alpha_iaa = 2.0

    # length hard constraint
    iqa_max_words = 65
    iaa_min_words = 125

    # task scaling
    iqa_scale = 1.0
    iaa_scale = 1.5

    # per-task threshold（也可以都用一个）
    thr_iqa = float((_GLOBAL_SCRIPT_ARGS.score_reward_threshold if _GLOBAL_SCRIPT_ARGS else 10.0))
    thr_iaa = float((_GLOBAL_SCRIPT_ARGS.score_reward_threshold if _GLOBAL_SCRIPT_ARGS else 10.0))

    # 把 threshold 映射成 sigma0，让 diff==threshold 时 reward==target
    # sigma0 = threshold / sqrt(-2 ln(target))
    def sigma0_from_thr(thr):
        return thr / math.sqrt(-2.0 * math.log(target))

    sigma0_iqa = sigma0_from_thr(thr_iqa)
    sigma0_iaa = sigma0_from_thr(thr_iaa)

    for t, content, gt in zip(tasks, contents, gts):
        pred = parse_rating_from_completion(content)
        if pred is None or gt is None:
            rewards.append(0.0)
            continue

        pred = float(pred)
        gt   = float(gt)
        pred = max(0.0, min(100.0, pred))
        gt   = max(0.0, min(100.0, gt))
        diff = abs(pred - gt)

        task_name = str(t).lower()
        is_iqa = ("iqa" in task_name)
        is_iaa = ("iaa" in task_name)

        # 1) 选任务参数
        if is_iqa:
            sigma0 = sigma0_iqa
            alpha = alpha_iqa
        else:
            sigma0 = sigma0_iaa
            alpha = alpha_iaa

        # 2) 自适应 sigma：sigma_dyn = sigma0 * (1 + alpha * diff/100)
        sigma_dyn = sigma0 * (1.0 + alpha * diff / 100.0)

        # 3) 高斯软奖励：exp(-diff^2 / (2*sigma_dyn^2))
        r = math.exp(-(diff * diff) / (2.0 * sigma_dyn * sigma_dyn))

        # 4) 长度硬惩罚（有reward才检查）
        if r > 0.0:
            words = len(content.split())
            if is_iqa and words > iqa_max_words:
                r = 0.0
            if is_iaa and words < iaa_min_words:
                r = 0.0

        # 5) 任务缩放（IAA更重要的话乘大点）
        if r > 0.0:
            r = r * (iqa_scale if is_iqa else iaa_scale)
            r = min(1.0, r)

        rewards.append(float(r))

    return rewards

# ----------------------- NEW: IAA Rank reward (continuous soft preference) -----------------------
def rank_reward(completions, solution, task, image_path, **kwargs):
    """
    只对 IAA 样本给 rank 奖励（IQA 返回 0）。

    核心思想：IAA 主观，不要把 “5.2 > 5.1” 当成绝对胜负。
    所以把 GT 分数差映射成“软偏好概率”，再让模型输出的分数差去拟合它。

    对 batch 中 IAA 样本集合：
      GT软偏好:   p_gt(i>j) = sigmoid((gt_i - gt_j)/tau)
      预测软偏好: p_pr(i>j) = sigmoid((pr_i - pr_j)/tau)

      Pair loss = BCE(p_gt, p_pr)
      难对(差值小)权重大: w_ij = exp(-|gt_i-gt_j|/m)

      对每个 i:
        mean_loss_i = sum_j w_ij * BCE / sum_j w_ij
        reward_i = 1 - mean_loss_i / max_bce
    """
    contents = [c[0]["content"] for c in completions]
    rewards = [0.0 for _ in contents]

    num_gen = len(task) // len(contents)
    tasks = task[::num_gen]
    gts   = solution[::num_gen]

    # ====== 超参（尽量少）======
    tau = 0.08      # soft preference 温度（0-1尺度）
    m   = 0.12      # hard-pair 权重温度（0-1尺度）
    eps = 1e-4      # BCE 稳定
    rank_scale = 1.0

    # 解析预测分数，并归一化到 0..1
    preds = []
    for content in contents:
        p = parse_rating_from_completion(content)
        preds.append(None if p is None else max(0.0, min(1.0, float(p) / 100.0)))

    # 收集 IAA index
    iaa_idx = []
    for i, t in enumerate(tasks):
        tn = str(t).lower()
        if ("iaa" in tn) and (preds[i] is not None) and (gts[i] is not None):
            iaa_idx.append(i)

    if len(iaa_idx) < 2:
        return rewards

    def sigmoid(x):
        return 1.0 / (1.0 + math.exp(-x))

    def bce(p, q):
        q = min(1.0 - eps, max(eps, q))
        return -(p * math.log(q) + (1.0 - p) * math.log(1.0 - q))

    max_bce = -math.log(eps)  # 大致上界，用来归一化

    # 对每个 IAA 样本 i 计算 reward_i
    for ii in iaa_idx:
        gt_i = float(gts[ii]) / 100.0  
        pr_i = float(preds[ii])        

        loss_sum = 0.0
        w_sum = 0.0

        for jj in iaa_idx:
            if jj == ii:
                continue
            gt_j = float(gts[jj]) / 100.0  
            pr_j = float(preds[jj])

            p_gt = sigmoid((gt_i - gt_j) / tau)
            p_pr = sigmoid((pr_i - pr_j) / tau) 

            w = math.exp(-abs(gt_i - gt_j) / m) 

            loss_sum += w * bce(p_gt, p_pr)
            w_sum += w

        mean_loss = loss_sum / w_sum
        r = 1.0 - (mean_loss / max_bce)
        r = max(0.0, min(1.0, r * rank_scale))

        rewards[ii] = float(r)

    return rewards
# ----------------------- NEW: IAA completion-level rank reward -----------------------
def rank_reward_thurstone(completions, solution, task, image_path, **kwargs):
    """
    Completion-level IAA rank reward.

    思想：
    1) 对同一样本的多个 completion，先估计 sample-level mean / variance
    2) 对于样本 x_i 的第 k 个 completion 分数 q_k(x_i)，
       用它去和其他样本的 mean score 比较，而不是复制一个 sample-level 常数 reward
    3) GT 偏好使用 soft preference:
           p_gt(i>j) = sigmoid((gt_i - gt_j) / tau)
    4) Pred 偏好使用 Thurstone-style:
           p_pred_k(i>j) = sigmoid((q_k(x_i) - mu_j) / sqrt(var_i + var_j + eps))
    5) Pair loss = BCE(p_gt, p_pred_k)
    6) 难对（gt差小）权重大:
           w_ij = exp(-|gt_i - gt_j| / m)
    7) 最终 reward 是 completion-level：
           r_k(i) = 1 - weighted_mean_BCE / max_bce

    只对 IAA 样本生效；IQA completion 返回 0.
    """

    contents = [c[0]["content"] for c in completions]
    rewards = [0.0 for _ in contents]

    # --------------------------------------------------
    # 1) 先对 metadata 做对齐
    # --------------------------------------------------
    if len(task) == len(contents):
        tasks = list(task)
        gts = list(solution)
        paths = list(image_path)
    else:
        num_gen = len(task) // len(contents)
        tasks = task[::num_gen]
        gts = solution[::num_gen]
        paths = image_path[::num_gen]

    # --------------------------------------------------
    # 2) 解析每个 completion 的预测分数，归一化到 [0,1]
    # --------------------------------------------------
    preds = []
    for content in contents:
        p = parse_rating_from_completion(content)
        preds.append(None if p is None else max(0.0, min(1.0, float(p) / 100.0)))

    # --------------------------------------------------
    # 3) 根据连续相同的 (task, gt, image_path) 把 completion 分成 sample group
    #    假设同一样本的多个 completion 是相邻排列的（GRPO 通常如此）
    # --------------------------------------------------
    group_ids = []
    group_keys = []
    current_gid = -1
    prev_key = None

    for t, gt, pth in zip(tasks, gts, paths):
        key = (str(t), float(gt) if gt is not None else None, str(pth))
        if key != prev_key:
            current_gid += 1
            group_keys.append(key)
            prev_key = key
        group_ids.append(current_gid)

    num_groups = current_gid + 1

    # --------------------------------------------------
    # 4) 统计每个 sample group 的 mean / variance
    # --------------------------------------------------
    group_means = [None for _ in range(num_groups)]
    group_vars  = [None for _ in range(num_groups)]
    group_tasks = [None for _ in range(num_groups)]
    group_gts   = [None for _ in range(num_groups)]

    for gid in range(num_groups):
        idxs = [i for i, g in enumerate(group_ids) if g == gid]
        vals = [preds[i] for i in idxs if preds[i] is not None]

        group_tasks[gid] = tasks[idxs[0]]
        group_gts[gid] = gts[idxs[0]]

        if len(vals) == 0:
            group_means[gid] = None
            group_vars[gid] = None
        else:
            mu = sum(vals) / len(vals)
            var = sum((x - mu) ** 2 for x in vals) / len(vals)
            group_means[gid] = mu
            group_vars[gid] = var

    # --------------------------------------------------
    # 5) 只收集 IAA group
    # --------------------------------------------------
    iaa_groups = []
    for gid in range(num_groups):
        task_name = str(group_tasks[gid]).lower()
        if ("iaa" in task_name) and (group_means[gid] is not None) and (group_gts[gid] is not None):
            iaa_groups.append(gid)

    if len(iaa_groups) < 2:
        return rewards

    # --------------------------------------------------
    # 6) 超参
    # --------------------------------------------------
    tau = 0.08      # soft preference 温度（GT差值 -> 概率）
    m = 0.12        # hard-pair 权重温度
    eps = 1e-6
    rank_scale = 1.0

    def sigmoid(x):
        return 1.0 / (1.0 + math.exp(-x))

    def bce(p, q):
        q = min(1.0 - eps, max(eps, q))
        return -(p * math.log(q) + (1.0 - p) * math.log(1.0 - q))

    max_bce = -math.log(eps)

    # --------------------------------------------------
    # 7) 对每个 completion 单独计算 reward
    # --------------------------------------------------
    for idx in range(len(contents)):
        gid_i = group_ids[idx]

        # 只对 IAA completion 做 rank reward
        if gid_i not in iaa_groups:
            continue

        q_i = preds[idx]
        if q_i is None:
            rewards[idx] = 0.0
            continue

        gt_i = float(group_gts[gid_i]) / 100.0
        var_i = group_vars[gid_i]

        loss_sum = 0.0
        w_sum = 0.0

        for gid_j in iaa_groups:
            if gid_j == gid_i:
                continue

            gt_j = float(group_gts[gid_j]) / 100.0
            mu_j = group_means[gid_j]
            var_j = group_vars[gid_j]

            # GT soft preference
            p_gt = sigmoid((gt_i - gt_j) / tau)

            # completion-level Thurstone-style predicted preference
            denom = math.sqrt(var_i + var_j + eps)
            p_pred = sigmoid((q_i - mu_j) / denom)

            # hard-pair weighting: gt差越小，权重越大
            w = math.exp(-abs(gt_i - gt_j) / m)

            loss_sum += w * bce(p_gt, p_pred)
            w_sum += w

        mean_loss = loss_sum / w_sum
        r = 1.0 - (mean_loss / max_bce)
        r = max(0.0, min(1.0, r * rank_scale))

        rewards[idx] = float(r)

    return rewards


def format_reward(completions, **kwargs):
    """
    Format reward:
      - Must contain <answer>...</answer>
      - Inside <answer> must be ONLY a single float (optionally with leading/trailing whitespace/newlines)
      - <think> is optional
    """
    rewards = []
    
    for completion in completions:
        content = completion[0]["content"]
        pred = parse_rating_from_completion(content)
        rewards.append(1.0 if pred is not None else 0.0)
    return rewards


reward_funcs_registry = {
    "accuracy": score_reward,
    "rank": rank_reward,
    "format": format_reward,
}


def main(script_args, training_args, model_args):
    global _GLOBAL_SCRIPT_ARGS
    _GLOBAL_SCRIPT_ARGS = script_args  # make threshold available to reward funcs

    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    print("reward_funcs:", reward_funcs)

    dataset = LazySupervisedDataset(script_args)

    trainer_cls = Qwen2VLGRPOTrainerUni
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        torch_dtype=model_args.torch_dtype,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)