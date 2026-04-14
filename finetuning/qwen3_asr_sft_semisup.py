"""Semi-supervised SFT script built on top of the stock ``qwen3_asr_sft.py``.

This file preserves the exact same multimodal training contract as the
supervised recipe:

- the prompt is still rendered with the Qwen3-ASR chat template,
- the real waveform is still loaded inside the collator,
- the target is still the decoder-side ASR answer suffix,
- prefix tokens are still masked out with ``-100``.

The only conceptual extension is that the training set may now contain multiple
sample sources with different trust levels:

1. Human-labeled supervised rows
2. Teacher-generated pseudo-labeled rows

Instead of branching the model or processor logic, we attach metadata
(``loss_weight`` and ``source``) to each row and change only the final loss
reduction. That keeps the semisupervised pipeline easy to reason about: the
model still solves the same task, but noisier examples count less.
"""

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
    """Build the minimal chat-template structure expected by the processor."""
    # This is intentionally the same message shape used in the supervised SFT
    # script so that both recipes share one prompt contract.
    return [
        {"role": "system", "content": prompt or ""},
        {"role": "user", "content": [{"type": "audio", "audio": audio_array}]},
    ]


def load_audio(path: str, sr: int = 16000):
    """Load one waveform as mono audio at the requested sample rate."""
    import librosa

    # Keeping waveform loading local to the collator avoids paying I/O cost
    # during dataset preprocessing and mirrors the stock SFT data flow.
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav


def make_preprocess_fn_prefix_only(processor):
    """Materialize prefix text and preserve per-sample loss weights.

    This mirrors the stock SFT recipe: prompt text is rendered once during
    dataset preprocessing, while the expensive audio loading stays inside the
    collator at batch time.
    """

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
            # ``loss_weight`` is preserved at row level so the trainer can later
            # decide how strongly each example should affect optimization.
            "loss_weight": float(ex.get("loss_weight", 1.0)),
            "source": str(ex.get("source", "supervised")),
        }

    return _preprocess


@dataclass
class DataCollatorForQwen3ASRSemiSup:
    processor: Any
    sampling_rate: int = 16000

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Build a multimodal batch and attach per-sample weights.

        Training principle:
            Like the stock script, this collator masks prompt-prefix tokens with
            ``-100`` so only the decoder-side transcription suffix contributes to
            the language-model loss.
        """
        # The only fields that change semantically versus the supervised collator
        # are the extra metadata tensors such as ``loss_weight``. The multimodal
        # prompt construction itself remains the same on purpose.
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

        # First pass: the real supervised sequence the model should learn to
        # generate after seeing the audio-conditioned prefix.
        full_inputs = self.processor(
            text=full_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        # Second pass: prefix-only encoding used strictly to recover the point
        # where the assistant answer begins in token space.
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
            # Exactly like the supervised recipe, the model should not be trained
            # to reproduce prompt scaffolding or audio placeholder structure.
            labels[i, :prefix_len] = -100

        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        full_inputs["labels"] = labels
        # ``loss_weight`` does not go into the model forward itself. It is
        # consumed later by the custom Trainer loss reduction.
        full_inputs["loss_weight"] = loss_weights
        return full_inputs


def build_trainer_class():
    """Create a Trainer subclass that supports sample-weighted LM loss.

    Defining the class inside a function avoids importing ``transformers.Trainer``
    at module import time, which keeps ``--help`` lightweight in partially
    provisioned environments.
    """
    from transformers import Trainer

    class WeightedCastFloatInputsTrainer(Trainer):
        """Trainer with float casting and sample-weighted language-model loss."""

        def _prepare_inputs(self, inputs):
            inputs = super()._prepare_inputs(inputs)
            model_dtype = getattr(self.model, "dtype", None)
            if model_dtype is not None:
                for key, value in list(inputs.items()):
                    if torch.is_tensor(value) and value.is_floating_point():
                        # This affects audio features and the extra weight tensor.
                        # Integer token ids remain untouched.
                        inputs[key] = value.to(dtype=model_dtype)
            return inputs

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            """Compute weighted autoregressive cross-entropy.

            The stock model already exposes a scalar ``loss``. We re-compute the
            token loss here only when sample weights are present so we can scale
            each sample before reducing across the batch.
            """
            loss_weight = inputs.pop("loss_weight", None)
            labels = inputs.get("labels")
            outputs = model(**inputs)

            if loss_weight is None or labels is None:
                # Fall back to the stock model loss when weights are absent.
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
                return (loss, outputs) if return_outputs else loss

            logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
            # We manually apply the usual causal-LM shift because we are
            # re-implementing the reduction step outside the model.
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
            # First average inside each sample so long utterances do not dominate
            # short ones purely because they have more supervised tokens.
            sample_losses = token_losses.sum(dim=1) / sample_token_count
            weights = loss_weight.to(sample_losses.device, dtype=sample_losses.dtype).clamp_min(0.0)
            denom = weights.sum().clamp_min(torch.finfo(sample_losses.dtype).eps)
            # Then form a weighted batch mean so pseudo labels can contribute more
            # gently than human-labeled rows.
            loss = (sample_losses * weights).sum() / denom
            return (loss, outputs) if return_outputs else loss

    return WeightedCastFloatInputsTrainer


def parse_args():
    """Define CLI arguments for mixed supervised/pseudo fine-tuning."""
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
    """Load one JSON/JSONL dataset split with the Hugging Face datasets library."""
    from datasets import load_dataset

    # The returned split is always named ``train`` here because we are loading a
    # single file at a time before later deciding how to combine it.
    ds = load_dataset("json", data_files={"train": path})
    return ds["train"]


def maybe_trim_dataset(ds, max_samples: int, seed: int):
    """Optionally downsample a dataset for ablations or smoke tests."""
    if max_samples <= 0 or len(ds) <= max_samples:
        return ds
    # Shuffle before select so truncation is not biased toward the file's
    # original ordering.
    return ds.shuffle(seed=seed).select(range(max_samples))


def attach_metadata(ds, default_weight: float, source: str):
    """Attach normalized metadata columns expected by the mixed-training pipeline."""
    def _map(ex):
        out = dict(ex)
        # Preserve row-level weights when the JSONL already contains them, but
        # provide a sensible default for plain supervised manifests.
        out["loss_weight"] = float(ex.get("loss_weight", default_weight))
        out["source"] = str(ex.get("source", source))
        out["prompt"] = str(ex.get("prompt", "") or "")
        return out

    return ds.map(_map, num_proc=1)


def build_train_dataset(args, processor):
    """Load, tag, concatenate and preprocess train datasets.

    Qwen3-ASR itself does not care whether a row came from human annotation or
    pseudo labeling. That distinction only matters for weighting and experiment
    analysis, so we encode it as metadata instead of branching the model code.
    """
    from datasets import concatenate_datasets

    supervised_path = args.supervised_train_file or args.train_file
    if not supervised_path and not args.pseudo_train_file:
        raise ValueError("Set --train_file/--supervised_train_file and/or --pseudo_train_file")

    pieces: List[Dataset] = []
    if supervised_path:
        # Supervised rows either inherit ``loss_weight=1.0`` or keep an explicit
        # row-level weight if one is already present in the JSONL.
        ds_sup = load_json_dataset(supervised_path)
        ds_sup = maybe_trim_dataset(ds_sup, args.max_supervised_samples, args.mix_seed)
        ds_sup = attach_metadata(ds_sup, args.supervised_loss_weight, "supervised")
        pieces.append(ds_sup)
        print(f"[dataset] supervised rows: {len(ds_sup)}")

    if args.pseudo_train_file:
        # Pseudo rows are structurally identical to supervised rows, which is why
        # simple concatenation works after metadata normalization.
        ds_pseudo = load_json_dataset(args.pseudo_train_file)
        ds_pseudo = maybe_trim_dataset(ds_pseudo, args.max_pseudo_samples, args.mix_seed)
        ds_pseudo = attach_metadata(ds_pseudo, args.pseudo_loss_weight, "pseudo")
        pieces.append(ds_pseudo)
        print(f"[dataset] pseudo rows:     {len(ds_pseudo)}")

    merged = pieces[0] if len(pieces) == 1 else concatenate_datasets(pieces)
    # Shuffle after concatenation so batches are not dominated by one source for
    # long stretches.
    merged = merged.shuffle(seed=args.mix_seed)
    processed = merged.map(make_preprocess_fn_prefix_only(processor), num_proc=1)
    keep = {"prompt", "audio", "target", "prefix_text", "loss_weight", "source"}
    drop = [col for col in processed.column_names if col not in keep]
    if drop:
        processed = processed.remove_columns(drop)
    print(f"[dataset] mixed train rows: {len(processed)}")
    return processed


def build_eval_dataset(args, processor):
    """Build the optional evaluation split using the same preprocessing recipe."""
    if not args.eval_file:
        return None
    from datasets import load_dataset

    # Eval rows keep the same tensor contract so the trainer can reuse the same
    # collator and model path with no special casing.
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
    """Run semi-supervised fine-tuning with a stock Qwen3-ASR checkpoint."""
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
    # As in the supervised recipe, we unwrap the convenience wrapper into the
    # actual HF model object plus the shared multimodal processor.
    model = asr_wrapper.model
    processor = asr_wrapper.processor

    patch_outer_forward(model)
    model.generation_config = GenerationConfig.from_model_config(model.config)

    # All dataset mixing happens outside the model. By the time batches reach
    # the network, both supervised and pseudo-labeled examples share the same
    # tensor contract.
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

    # Resume behavior intentionally matches the supervised script so experiments
    # can switch between recipes without changing checkpoint management habits.
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
