"""Semi-supervised SFT script built on top of the stock qwen3_asr_sft.py."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def build_prefix_messages(prompt: str, audio_array):
    return [
        {"role": "system", "content": prompt or ""},
        {"role": "user", "content": [{"type": "audio", "audio": audio_array}]},
    ]


def load_audio(path: str, sr: int = 16000):
    import librosa

    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav


def make_preprocess_fn_prefix_only(processor):
    """Materialize prefix text and preserve per-sample loss weights."""

    def _preprocess(ex: Dict[str, Any]) -> Dict[str, Any]:
        prompt = ex.get("prompt", "")
        prefix_msgs = build_prefix_messages(prompt, None)
        prefix_text = processor.apply_chat_template(
            [prefix_msgs],
            add_generation_prompt=True,
            tokenize=False,
        )[0]
        return {
            "prompt": prompt,
            "audio": ex["audio"],
            "target": ex["text"],
            "prefix_text": prefix_text,
            "loss_weight": float(ex.get("loss_weight", 1.0)),
            "source": str(ex.get("source", "supervised")),
        }

    return _preprocess


@dataclass
class DataCollatorForQwen3ASRSemiSup:
    processor: Any
    sampling_rate: int = 16000

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audio_paths = [f["audio"] for f in features]
        prefix_texts = [f["prefix_text"] for f in features]
        targets = [f["target"] for f in features]
        loss_weights = torch.tensor(
            [float(f.get("loss_weight", 1.0)) for f in features],
            dtype=torch.float32,
        )

        eos = self.processor.tokenizer.eos_token or ""
        full_texts = [pfx + tgt + eos for pfx, tgt in zip(prefix_texts, targets)]
        audios = [load_audio(path, sr=self.sampling_rate) for path in audio_paths]

        full_inputs = self.processor(
            text=full_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        prefix_inputs = self.processor(
            text=prefix_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )

        prefix_lens = prefix_inputs["attention_mask"].sum(dim=1).tolist()
        labels = full_inputs["input_ids"].clone()
        for i, prefix_len in enumerate(prefix_lens):
            labels[i, :prefix_len] = -100

        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        full_inputs["labels"] = labels
        full_inputs["loss_weight"] = loss_weights
        return full_inputs


def build_trainer_class():
    from transformers import Trainer

    class WeightedCastFloatInputsTrainer(Trainer):
        """Trainer with float casting and sample-weighted language-model loss."""

        def _prepare_inputs(self, inputs):
            inputs = super()._prepare_inputs(inputs)
            model_dtype = getattr(self.model, "dtype", None)
            if model_dtype is not None:
                for key, value in list(inputs.items()):
                    if torch.is_tensor(value) and value.is_floating_point():
                        inputs[key] = value.to(dtype=model_dtype)
            return inputs

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            loss_weight = inputs.pop("loss_weight", None)
            labels = inputs.get("labels")
            outputs = model(**inputs)

            if loss_weight is None or labels is None:
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
                return (loss, outputs) if return_outputs else loss

            logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            vocab_size = shift_logits.size(-1)

            token_losses = F.cross_entropy(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="none",
            ).view_as(shift_labels)
            token_mask = shift_labels.ne(-100)
            sample_token_count = token_mask.sum(dim=1).clamp_min(1)
            sample_losses = token_losses.sum(dim=1) / sample_token_count
            weights = loss_weight.to(sample_losses.device, dtype=sample_losses.dtype).clamp_min(0.0)
            denom = weights.sum().clamp_min(torch.finfo(sample_losses.dtype).eps)
            loss = (sample_losses * weights).sum() / denom
            return (loss, outputs) if return_outputs else loss

    return WeightedCastFloatInputsTrainer


def parse_args():
    p = argparse.ArgumentParser("Qwen3-ASR Semi-Supervised Finetuning")

    p.add_argument("--model_path", type=str, default="Qwen/Qwen3-ASR-1.7B")
    p.add_argument("--train_file", type=str, default="", help="Backward-compatible alias for supervised train JSONL")
    p.add_argument("--eval_file", type=str, default="")
    p.add_argument("--output_dir", type=str, default="./qwen3-asr-semisup-out")

    p.add_argument("--supervised_train_file", type=str, default="")
    p.add_argument("--pseudo_train_file", type=str, default="")
    p.add_argument("--supervised_loss_weight", type=float, default=1.0)
    p.add_argument("--pseudo_loss_weight", type=float, default=0.3)
    p.add_argument("--max_supervised_samples", type=int, default=0)
    p.add_argument("--max_pseudo_samples", type=int, default=0)
    p.add_argument("--mix_seed", type=int, default=42)

    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--grad_acc", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--epochs", type=float, default=1)
    p.add_argument("--log_steps", type=int, default=10)
    p.add_argument("--lr_scheduler_type", type=str, default="linear")
    p.add_argument("--warmup_ratio", type=float, default=0.02)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pin_memory", type=int, default=1)
    p.add_argument("--persistent_workers", type=int, default=1)
    p.add_argument("--prefetch_factor", type=int, default=2)
    p.add_argument("--save_strategy", type=str, default="steps")
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--save_total_limit", type=int, default=5)
    p.add_argument("--resume_from", type=str, default="")
    p.add_argument("--resume", type=int, default=0)
    return p.parse_args()


def load_json_dataset(path: str):
    from datasets import load_dataset

    ds = load_dataset("json", data_files={"train": path})
    return ds["train"]


def maybe_trim_dataset(ds, max_samples: int, seed: int):
    if max_samples <= 0 or len(ds) <= max_samples:
        return ds
    return ds.shuffle(seed=seed).select(range(max_samples))


def attach_metadata(ds, default_weight: float, source: str):
    def _map(ex):
        out = dict(ex)
        out["loss_weight"] = float(ex.get("loss_weight", default_weight))
        out["source"] = str(ex.get("source", source))
        out["prompt"] = str(ex.get("prompt", "") or "")
        return out

    return ds.map(_map, num_proc=1)


def build_train_dataset(args, processor):
    from datasets import concatenate_datasets

    supervised_path = args.supervised_train_file or args.train_file
    if not supervised_path and not args.pseudo_train_file:
        raise ValueError("Set --train_file/--supervised_train_file and/or --pseudo_train_file")

    pieces: List[Dataset] = []
    if supervised_path:
        ds_sup = load_json_dataset(supervised_path)
        ds_sup = maybe_trim_dataset(ds_sup, args.max_supervised_samples, args.mix_seed)
        ds_sup = attach_metadata(ds_sup, args.supervised_loss_weight, "supervised")
        pieces.append(ds_sup)
        print(f"[dataset] supervised rows: {len(ds_sup)}")

    if args.pseudo_train_file:
        ds_pseudo = load_json_dataset(args.pseudo_train_file)
        ds_pseudo = maybe_trim_dataset(ds_pseudo, args.max_pseudo_samples, args.mix_seed)
        ds_pseudo = attach_metadata(ds_pseudo, args.pseudo_loss_weight, "pseudo")
        pieces.append(ds_pseudo)
        print(f"[dataset] pseudo rows:     {len(ds_pseudo)}")

    merged = pieces[0] if len(pieces) == 1 else concatenate_datasets(pieces)
    merged = merged.shuffle(seed=args.mix_seed)
    processed = merged.map(make_preprocess_fn_prefix_only(processor), num_proc=1)
    keep = {"prompt", "audio", "target", "prefix_text", "loss_weight", "source"}
    drop = [col for col in processed.column_names if col not in keep]
    if drop:
        processed = processed.remove_columns(drop)
    print(f"[dataset] mixed train rows: {len(processed)}")
    return processed


def build_eval_dataset(args, processor):
    if not args.eval_file:
        return None
    from datasets import load_dataset

    ds_eval = load_dataset("json", data_files={"validation": args.eval_file})["validation"]
    ds_eval = attach_metadata(ds_eval, 1.0, "eval")
    ds_eval = ds_eval.map(make_preprocess_fn_prefix_only(processor), num_proc=1)
    keep = {"prompt", "audio", "target", "prefix_text", "loss_weight", "source"}
    drop = [col for col in ds_eval.column_names if col not in keep]
    if drop:
        ds_eval = ds_eval.remove_columns(drop)
    print(f"[dataset] eval rows:       {len(ds_eval)}")
    return ds_eval


def main():
    args = parse_args()
    from qwen3_asr_sft import (
        MakeEveryCheckpointInferableCallback,
        find_latest_checkpoint,
        patch_outer_forward,
    )
    from qwen_asr import Qwen3ASRModel
    from transformers import GenerationConfig, TrainingArguments

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8

    asr_wrapper = Qwen3ASRModel.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16 if use_bf16 else torch.float16,
        device_map=None,
    )
    model = asr_wrapper.model
    processor = asr_wrapper.processor

    patch_outer_forward(model)
    model.generation_config = GenerationConfig.from_model_config(model.config)

    train_dataset = build_train_dataset(args, processor)
    eval_dataset = build_eval_dataset(args, processor)
    collator = DataCollatorForQwen3ASRSemiSup(processor=processor, sampling_rate=args.sr)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=args.log_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=(args.pin_memory == 1),
        dataloader_persistent_workers=(args.persistent_workers == 1),
        dataloader_prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_safetensors=True,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        do_eval=bool(eval_dataset),
        bf16=use_bf16,
        fp16=not use_bf16,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        report_to="none",
    )

    WeightedCastFloatInputsTrainer = build_trainer_class()
    trainer = WeightedCastFloatInputsTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        tokenizer=processor.tokenizer,
        callbacks=[MakeEveryCheckpointInferableCallback(base_model_path=args.model_path)],
    )

    resume_from = (args.resume_from or "").strip()
    if not resume_from and args.resume == 1:
        resume_from = find_latest_checkpoint(training_args.output_dir) or ""

    if resume_from:
        if trainer.args.process_index == 0:
            print(f"[resume] resume_from_checkpoint = {resume_from}")
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()


if __name__ == "__main__":
    main()
