# Copyright 2025
# Stage-1 SFT for unified IQA/IAA (Qwen2.5-VL) with TRL SFTTrainer
# - Reads two jsonl files directly (IQA + IAA) with CoT
# - Prompts aligned with your GRPO stage (same SYSTEM/IQA/IAA prompts)
# - Mixes tasks via interleave_datasets (ratio controlled)
# - Rewrites <answer> to 0-100 scale (IMPORTANT)
# - Assistant-only loss (mask system+user prompt tokens), mask pad + image tokens in labels
# - Uses AutoProcessor(min_pixels/max_pixels) like GRPO processing_class setup

import os
import re
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional

import torch
from PIL import Image
from datasets import load_dataset, interleave_datasets

from transformers import AutoProcessor, set_seed
from trl import (
    SFTTrainer,
    SFTConfig,
    ModelConfig,
    TrlParser,
    get_peft_config,
    get_kbit_device_map,
    get_quantization_config,
)
from trl.data_utils import maybe_apply_chat_template
# =========================================================
# Prompts (对齐 GRPO)
# =========================================================
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it.\n"
    "The Assistant may include optional reasoning inside <think>...</think> (it may be brief or detailed).\n"
    "The final answer MUST be provided inside <answer>...</answer>.\n"
    "Inside <answer>, output ONLY a single numeric rating (a float), with no extra text.\n"
    "If <think> is present, it must appear before <answer>."
)

IQA_PROMPT = (
    "TASK: Image Quality Assessment (IQA).\n"
    "What is your overall rating on the quality of the given picture?\n"
    "The rating should be a float between 0 and 100, rounded to two decimal places, "
    "with 0 representing very poor quality and 100 representing excellent quality.\n"
    "You may answer briefly without detailed reasoning.\n"
    "In <answer>, output ONLY the numeric score (e.g., 73.42)."
)

IAA_PROMPT = (
    "TASK: Image Aesthetic Assessment (IAA).\n"
    "What is your overall rating on the aesthetic quality of the given picture?\n"
    "The rating should be a float between 0 and 100, rounded to two decimal places, "
    "with 0 representing very poor aesthetics and 100 representing excellent aesthetics.\n"
    "Consider composition, lighting, color harmony, subject presentation, balance, style, and emotional impact.\n"
    "In <answer>, output ONLY the numeric score (e.g., 73.42)."
)

# =========================================================
# Script Args（只放数据与混合相关参数）
# =========================================================
@dataclass
class SFTScriptArguments:
    # 必填（无 default 的字段必须放前面）
    iqa_jsonl: str = field(metadata={"help": "Path to IQA jsonl (fields: image, gt_score, reversal_thinking)"})
    iaa_jsonl: str = field(metadata={"help": "Path to IAA jsonl (fields: image, gt_score, cot_reasoning)"})
    image_root: str = field(metadata={"help": "Root directory of images"})

    # 可选
    mix_prob_iqa: float = field(default=0.5, metadata={"help": "Sampling probability for IQA in interleave"})
    stopping_strategy: str = field(
        default="all_exhausted",
        metadata={"help": "interleave_datasets stopping_strategy: all_exhausted | first_exhausted"},
    )

    # 对齐 GRPO 的视觉预算
    max_pixels: int = field(default=786432, metadata={"help": "Align with GRPO --max_pixels"})
    min_pixels: int = field(default=3136, metadata={"help": "Align with GRPO --min_pixels"})


# =========================================================
# 分数映射 & <answer> 重写（关键：统一到 0-100）
# =========================================================
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)

def gt_to_100(task: str, gt_score: float) -> float:
    """
    Match your GRPO gt_to_100 mapping logic.
    - IQA: gt in [0,1] or [1,5] or already [0,100]
    - IAA: gt in [0,1] or [1,10] or already [0,100]
    """
    x = float(gt_score)
    if task == "iqa":
        if 0.0 <= x <= 1.0:
            return x * 100.0
        if 1.0 <= x <= 5.0:
            return (x - 1.0) / 4.0 * 100.0
        return max(0.0, min(100.0, x))
    else:  # iaa
        if 0.0 <= x <= 10.0:
            return x * 10.0
        return max(0.0, min(100.0, x))

def rewrite_answer(cot: str, score_100: float) -> str:
    """
    Replace <answer>...</answer> with <answer>xx.xx</answer> on 0-100 scale.
    """
    score_str = f"{float(score_100):.2f}"
    cot = (cot or "").strip()
    if ANSWER_RE.search(cot):
        return ANSWER_RE.sub(f"<answer>{score_str}</answer>", cot)
    return cot + f"\n<answer>{score_str}</answer>"

def resolve_image(task: str, root: str, rel: str) -> str:
    if os.path.isabs(rel):
        return rel
    if task == "iqa":
        return os.path.join(root, "Data-DeQA-Score", rel)
    return os.path.join(root, "ArtiMuse-10K-251219/images", rel)

def build_messages(task: str, cot: str) -> List[Dict[str, Any]]:
    prompt = IQA_PROMPT if task == "iqa" else IAA_PROMPT
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
        {"role": "assistant", "content": [{"type": "text", "text": cot}]},
    ]


# =========================================================
# Collator（assistant-only loss + mask image token）
# =========================================================
def _get_image_token_id(processor) -> Optional[int]:
    # Qwen VL processor typically has processor.image_token (a string token like "<image>")
    if hasattr(processor, "image_token"):
        try:
            tid = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
            if isinstance(tid, int) and tid >= 0:
                return tid
        except Exception:
            pass
    return None

def make_collate_fn(processor):
    image_token_id = _get_image_token_id(processor)

    def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        prompt_texts: List[str] = []
        full_texts: List[str] = []
        images: List[Image.Image] = []

        for ex in examples:
            msgs_full = ex["messages"]
            msgs_prompt = msgs_full[:-1]

            prompt_txt = maybe_apply_chat_template({"prompt": msgs_prompt}, processor)["prompt"]
            full_txt   = maybe_apply_chat_template({"prompt": msgs_full}, processor)["prompt"]

            prompt_texts.append(prompt_txt)
            full_texts.append(full_txt)

            with Image.open(ex["image_path"]) as im:
                images.append(im.convert("RGB"))

        batch = processor(
            text=full_texts,
            images=images,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            truncation=True,
            add_special_tokens=False,
        )
        batch_prompt = processor(
            text=prompt_texts,
            images=images,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            truncation=True,
            add_special_tokens=False,
        )

        labels = batch["input_ids"].clone()

        # mask system+user (prompt) tokens -> assistant-only loss
        for i in range(labels.size(0)):
            prompt_len = int(batch_prompt["attention_mask"][i].sum().item())
            labels[i, :prompt_len] = -100

        # mask pad
        pad_id = processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        # mask image token if exists
        if image_token_id is not None:
            labels[labels == image_token_id] = -100

        batch["labels"] = labels
        return batch

    return collate_fn


# =========================================================
# Main
# =========================================================
def main(script_args: SFTScriptArguments, training_args: SFTConfig, model_args: ModelConfig):
    set_seed(training_args.seed)

    # Load jsonl datasets
    ds_iqa = load_dataset("json", data_files=script_args.iqa_jsonl, split="train")
    ds_iaa = load_dataset("json", data_files=script_args.iaa_jsonl, split="train")

    def convert_iqa(ex: Dict[str, Any]) -> Dict[str, Any]:
        cot = ex.get("reversal_thinking", "")
        score_raw = float(ex["gt_score"])
        score_100 = gt_to_100("iqa", score_raw)
        cot = rewrite_answer(cot, score_100)
        return {
            "messages": build_messages("iqa", cot),
            "image_path": resolve_image("iqa", script_args.image_root, ex["image"]),
        }

    def convert_iaa(ex: Dict[str, Any]) -> Dict[str, Any]:
        cot = ex.get("cot_reasoning", "")
        score_raw = float(ex["gt_score"])
        score_100 = gt_to_100("iaa", score_raw)
        cot = rewrite_answer(cot, score_100)
        return {
            "messages": build_messages("iaa", cot),
            "image_path": resolve_image("iaa", script_args.image_root, ex["image"]),
        }

    ds_iqa = ds_iqa.map(convert_iqa, remove_columns=ds_iqa.column_names, desc="convert IQA")
    ds_iaa = ds_iaa.map(convert_iaa, remove_columns=ds_iaa.column_names, desc="convert IAA")

    # Mix datasets (within-batch mixing by interleaving samples)
    p_iqa = float(script_args.mix_prob_iqa)
    train_dataset = interleave_datasets(
        [ds_iqa, ds_iaa],
        probabilities=[p_iqa, 1.0 - p_iqa],
        seed=training_args.seed,
        stopping_strategy=script_args.stopping_strategy,
    )

    # Processor (对齐 GRPO：设置 min/max_pixels)
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        min_pixels=script_args.min_pixels,
        max_pixels=script_args.max_pixels,
    )
    # pad token
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # Model init kwargs
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    quant_cfg = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quant_cfg is not None else None,
        quantization_config=quant_cfg,
    )

    from transformers import Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs,
    )

    # TRL/Trainer housekeeping for custom collator
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}
    training_args.remove_unused_columns = False

    collate_fn = make_collate_fn(processor)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=collate_fn,
        processing_class=processor,
        peft_config=get_peft_config(model_args),
    )

    trainer.train()

    # 只保存模型（避免你之前 preprocessor_config.json 写权限问题时卡住）
    trainer.save_model(training_args.output_dir)
    if trainer.accelerator.is_main_process:
        
        # 1. 修正 config.json (你原本写的)
        trainer.model.config.use_cache = True
        
        # 2. 【新增】修正 generation_config.json
        if trainer.model.generation_config is not None:
            trainer.model.generation_config.use_cache = True
            trainer.model.generation_config.save_pretrained(training_args.output_dir)
            
        # 3. 保存 config.json
        trainer.model.config.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)