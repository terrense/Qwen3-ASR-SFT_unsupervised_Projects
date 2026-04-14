# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
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
"""
Supervised fine-tuning script for Qwen3-ASR.

How to read this file conceptually:

1. A training row starts life as ``{"audio": ..., "text": ..., "prompt": ...}``.
2. We render the chat-template prefix once with ``apply_chat_template(...)`` and
   store that textual prefix as ``prefix_text``.
3. At batch time we load the real waveform, concatenate
   ``prefix_text + target_text + eos``, and run the multimodal processor.
4. We mask the prefix tokens in ``labels`` with ``-100`` so only the decoder-side
   ASR answer contributes to the loss.
5. Hugging Face ``Trainer`` then performs standard causal-LM optimization on
   those unmasked target positions.

The key design choice is that this script preserves the same multimodal prompt
contract used at inference time. We are not training a separate ASR head with a
CTC objective; we are continuing to teach the released Qwen3-ASR model how to
generate the expected ASR answer format after seeing an audio-conditioned prompt.
"""

import argparse
import os
import re
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import librosa
import torch
from datasets import load_dataset
from qwen_asr import Qwen3ASRModel
from transformers import (GenerationConfig, Trainer, TrainerCallback,
                          TrainingArguments)


def patch_outer_forward(model):
    """
    Patch the outer model ``forward`` so Hugging Face Trainer can call it directly.

    The public wrapper stores the real multimodal logic under ``model.thinker``.
    Trainer, however, expects ``model.forward`` to consume a batch dictionary and
    return a standard output object, so we proxy the call.
    """
    # The checkpoint loaded by ``Qwen3ASRModel.from_pretrained`` exposes an outer
    # Hugging Face module object whose real multimodal forward path lives under
    # ``model.thinker``. Generation works through that wrapper already, but
    # training with ``Trainer`` expects a callable ``model.forward(**batch)`` on
    # the outer object. This patch gives Trainer exactly that entrypoint.
    #
    # We patch the *class* instead of only one bound instance method because
    # Hugging Face wrappers may copy or rebind the model object during training.
    # A class-level patch keeps the behavior stable across those code paths.
    cls = model.__class__
    if getattr(cls, "_forward_patched", False):
        return

    if not hasattr(model, "thinker") or not hasattr(model.thinker, "forward"):
        raise RuntimeError(
            "Cannot patch forward: model has no `.thinker.forward`. "
            "Your qwen3_asr model may be incompatible."
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        input_features=None,
        feature_attention_mask=None,
        labels=None,
        **kwargs,
    ):
        # Keep the signature aligned with Trainer batch dictionaries. ``**kwargs``
        # makes the shim forward-compatible with extra tensors added upstream.
        #
        # The collator in this file emits the same multimodal tensor keys the
        # thinker expects:
        # - ``input_ids`` / ``attention_mask`` for the textual prompt side
        # - ``input_features`` / ``feature_attention_mask`` for the audio encoder
        # - ``labels`` for autoregressive supervision
        return self.thinker.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            labels=labels,
            **kwargs,
        )

    cls.forward = forward
    cls._forward_patched = True


_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    """Return the numerically latest ``checkpoint-*`` directory, if any."""
    if not output_dir or not os.path.isdir(output_dir):
        return None
    best_step = None
    best_path = None
    for name in os.listdir(output_dir):
        m = _CKPT_RE.match(name)
        if not m:
            continue
        # Numeric comparison is intentional: lexicographic ordering would treat
        # ``checkpoint-9`` as newer than ``checkpoint-100``.
        step = int(m.group(1))
        path = os.path.join(output_dir, name)
        if os.path.isdir(path) and (best_step is None or step > best_step):
            best_step = step
            best_path = path
    return best_path


def load_audio(path: str, sr: int = 16000):
    """Load one training waveform as mono audio at the target sample rate."""
    # Using the same 16 kHz mono convention as inference keeps feature extraction
    # consistent between training and deployment.
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav


def build_prefix_messages(prompt: str, audio_array):
    """Build the chat-style message skeleton used for prefix-only loss masking."""
    # This mirrors the structure used by the inference wrapper:
    # - ``system`` carries optional task instruction / context
    # - ``user`` carries a multimodal content list with an ``audio`` slot
    #
    # The actual waveform may be omitted during cheap preprocessing because the
    # chat template only needs to know that an audio segment will appear here.
    return [
        {"role": "system", "content": prompt or ""},
        {"role": "user", "content": [{"type": "audio", "audio": audio_array}]},
    ]


def make_preprocess_fn_prefix_only(processor):
    """
    Create a dataset-map function that materializes the prompt prefix text.

    We precompute the chat-template prefix once per example so the collator only
    needs to load audio and concatenate the target transcription at batch time.
    """
    def _preprocess(ex: Dict[str, Any]) -> Dict[str, Any]:
        prompt = ex.get("prompt", "")
        dummy_audio = None
        # The chat template only needs an "audio slot" in the message structure
        # to render the textual prefix. Actual waveform loading stays in the
        # collator so this preprocessing step remains cheap.
        prefix_msgs = build_prefix_messages(prompt, dummy_audio)
        prefix_text = processor.apply_chat_template(
            [prefix_msgs], add_generation_prompt=True, tokenize=False
        )[0]
        # The dataset keeps only the minimum information the collator needs later:
        # - the raw audio path
        # - the target ASR text (already normalized by data prep)
        # - the rendered textual prefix that should *not* contribute to the loss
        return {
            "prompt": prompt,
            "audio": ex["audio"],
            "target": ex["text"],
            "prefix_text": prefix_text,
        }

    return _preprocess


@dataclass
class DataCollatorForQwen3ASRFinetuning:
    processor: Any
    sampling_rate: int = 16000

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Build a training batch and mask out non-target tokens in ``labels``.

        AI note:
            ASR fine-tuning is framed as conditional generation. The model should
            learn to predict the transcription, not the prompt prefix nor padding
            tokens, so those label positions are set to ``-100``.
        """
        # Each feature is still a lightweight Python dict from the dataset map
        # stage. The collator is the first place where we actually touch the
        # audio files and turn everything into tensors.
        audio_paths = [f["audio"] for f in features]
        prefix_texts = [f["prefix_text"] for f in features]
        targets = [f["target"] for f in features]

        # Adding EOS gives the decoder an explicit supervised stop signal.
        eos = self.processor.tokenizer.eos_token or ""
        full_texts = [pfx + tgt + eos for pfx, tgt in zip(prefix_texts, targets)]
        audios = [load_audio(p, sr=self.sampling_rate) for p in audio_paths]

        # ``full_inputs`` is the tensorized form of the *entire* teacher-forced
        # sequence: chat prefix + expected ASR answer + EOS.
        full_inputs = self.processor(
            text=full_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        # ``prefix_inputs`` is a second processor pass used only for bookkeeping.
        # We intentionally feed the same audio here because we want the token
        # boundary inside the exact multimodal sequence that the model will see.
        #
        # Why not estimate the prefix length from raw string length?
        # Because tokenization, special tokens, and audio placeholder expansion
        # are handled inside the processor. Asking the processor directly is the
        # safest way to recover the true answer-start position.
        prefix_inputs = self.processor(
            text=prefix_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )

        # ``prefix_inputs`` tells us where the supervision should start.
        # Summing the attention mask counts real tokens correctly regardless of
        # whether the tokenizer pads on the left or the right.
        prefix_lens = prefix_inputs["attention_mask"].sum(dim=1).tolist()
        labels = full_inputs["input_ids"].clone()
        for i, pl in enumerate(prefix_lens):
            # HF loss helpers ignore ``-100`` positions when computing
            # cross-entropy, so the prompt prefix does not contribute to the loss.
            #
            # After this assignment, the model is optimized only on the target
            # suffix, not on the structural prompt tokens that set up the task.
            labels[i, :pl] = -100

        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            # Padding is also masked because it is an artifact of batching, not
            # something the model should try to predict.
            labels[labels == pad_id] = -100

        # Trainer forwards this to ``model(..., labels=labels)``, and the model's
        # loss function handles the next-token shift internally.
        full_inputs["labels"] = labels
        return full_inputs


class CastFloatInputsTrainer(Trainer):
    """Trainer that casts floating-point tensors to the model's compute dtype."""

    def _prepare_inputs(self, inputs):
        inputs = super()._prepare_inputs(inputs)
        model_dtype = getattr(self.model, "dtype", None)
        if model_dtype is not None:
            for k, v in list(inputs.items()):
                if torch.is_tensor(v) and v.is_floating_point():
                    # Audio features are floating tensors; matching the model
                    # dtype avoids unnecessary mixed-precision casts later.
                    # Integer tensors such as ``input_ids`` are intentionally
                    # left untouched.
                    inputs[k] = v.to(dtype=model_dtype)
        return inputs


def copy_required_hf_files_for_qwen_asr(src_dir: str, dst_dir: str):
    """Copy tokenizer/processor metadata so each checkpoint remains directly loadable."""
    os.makedirs(dst_dir, exist_ok=True)
    # Multimodal checkpoints are not self-describing from weights alone. The
    # processor config, tokenizer vocab and chat template are part of inference.
    #
    # Without these sidecar files, a saved ``checkpoint-*`` directory might have
    # the fine-tuned weights but still be impossible to reload correctly through
    # ``Qwen3ASRModel.from_pretrained(...)`` or Hugging Face auto classes.
    required = [
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "chat_template.json",
        "merges.txt",
        "vocab.json",
    ]
    for fn in required:
        src = os.path.join(src_dir, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst_dir, fn))


class MakeEveryCheckpointInferableCallback(TrainerCallback):
    """Ensure saved checkpoints contain the non-weight files needed for inference."""

    def __init__(self, base_model_path: str):
        self.base_model_path = base_model_path

    def on_save(self, args: TrainingArguments, state, control, **kwargs):
        if args.process_index != 0:
            # In distributed training only rank 0 performs the metadata copy to
            # avoid redundant writes to the same checkpoint directory.
            return control

        # Different Trainer save code paths expose the checkpoint path in
        # slightly different ways, so we support both conventions here.
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.isdir(ckpt_dir):
            ckpt_dir = kwargs.get("checkpoint", ckpt_dir)

        copy_required_hf_files_for_qwen_asr(self.base_model_path, ckpt_dir)
        return control


def parse_args():
    """Parse CLI arguments for fine-tuning and checkpoint-resume behavior."""
    p = argparse.ArgumentParser("Qwen3-ASR Finetuning")

    # Paths
    p.add_argument("--model_path", type=str, default="Qwen/Qwen3-ASR-1.7B")
    p.add_argument("--train_file", type=str, default="train.jsonl")
    p.add_argument("--eval_file", type=str, default="")
    p.add_argument("--output_dir", type=str, default="./qwen3-asr-finetuning-out")

    # Audio
    p.add_argument("--sr", type=int, default=16000)

    # Train hyper-params
    # These correspond closely to ``TrainingArguments`` fields configured below.
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--grad_acc", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--epochs", type=float, default=1)
    p.add_argument("--log_steps", type=int, default=10)
    p.add_argument("--lr_scheduler_type", type=str, default="linear")
    p.add_argument("--warmup_ratio", type=float, default=0.02)

    # DataLoader
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pin_memory", type=int, default=1)
    p.add_argument("--persistent_workers", type=int, default=1)
    p.add_argument("--prefetch_factor", type=int, default=2)

    # Save
    p.add_argument("--save_strategy", type=str, default="steps")
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--save_total_limit", type=int, default=5)

    # Resume
    # ``resume`` is a script-friendly boolean flag, while ``resume_from`` allows
    # a precise checkpoint path override.
    p.add_argument("--resume_from", type=str, default="")
    p.add_argument("--resume", type=int, default=0)

    return p.parse_args()


def main():
    """Run dataset loading, batch construction and Hugging Face training."""
    args_cli = parse_args()

    if not args_cli.train_file:
        raise ValueError("TRAIN_FILE is required (json/jsonl). Needs fields: audio, text, optional prompt")

    # BF16 is preferred on newer accelerators because it keeps a wide exponent
    # range with lower memory cost than FP32. Older GPUs fall back to FP16.
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    asr_wrapper = Qwen3ASRModel.from_pretrained(
        args_cli.model_path,
        dtype=torch.bfloat16 if use_bf16 else torch.float16,
        device_map=None,
    )
    # ``Qwen3ASRModel`` is a convenience wrapper used heavily by inference code.
    # For SFT we only need the underlying HF model plus the shared multimodal
    # processor, so we unwrap them immediately.
    model = asr_wrapper.model
    processor = asr_wrapper.processor

    # Trainer talks to the outer HF model object, so patch it before training.
    patch_outer_forward(model)
    model.generation_config = GenerationConfig.from_model_config(model.config)

    raw_ds = load_dataset(
        "json",
        data_files={
            "train": args_cli.train_file,
            **({"validation": args_cli.eval_file} if args_cli.eval_file else {}),
        },
    )
    # ``make_preprocess_fn_prefix_only`` is a closure that captures ``processor``
    # and returns a dataset-mapping function.
    ds = raw_ds.map(make_preprocess_fn_prefix_only(processor), num_proc=1)

    keep = {"prompt", "audio", "target", "prefix_text"}
    for split in ds.keys():
        # Drop unused JSON columns early so the dataloader only carries the
        # fields required by the collator. This keeps dataset examples easy to
        # inspect and avoids accidentally carrying unrelated annotation columns
        # through the whole training pipeline.
        drop = [c for c in ds[split].column_names if c not in keep]
        if drop:
            ds[split] = ds[split].remove_columns(drop)

    # From this point onward the data flow is:
    # dataset example -> collator builds tensors -> trainer calls model.forward
    # -> model computes autoregressive loss on the unmasked target suffix.
    collator = DataCollatorForQwen3ASRFinetuning(processor=processor, sampling_rate=args_cli.sr)

    training_args = TrainingArguments(
        output_dir=args_cli.output_dir,
        per_device_train_batch_size=args_cli.batch_size,
        gradient_accumulation_steps=args_cli.grad_acc,
        learning_rate=args_cli.lr,
        num_train_epochs=args_cli.epochs,
        logging_steps=args_cli.log_steps,
        lr_scheduler_type=args_cli.lr_scheduler_type,
        warmup_ratio=args_cli.warmup_ratio,
        dataloader_num_workers=args_cli.num_workers,
        dataloader_pin_memory=(args_cli.pin_memory == 1),
        dataloader_persistent_workers=(args_cli.persistent_workers == 1),
        dataloader_prefetch_factor=args_cli.prefetch_factor if args_cli.num_workers > 0 else None,
        save_strategy=args_cli.save_strategy,
        save_steps=args_cli.save_steps,
        save_total_limit=args_cli.save_total_limit,
        save_safetensors=True,
        # Evaluate on the same cadence as checkpointing so each saved model has a
        # nearby validation snapshot.
        eval_strategy="steps",
        eval_steps=args_cli.save_steps,
        do_eval=bool(args_cli.eval_file),
        bf16=use_bf16,
        fp16=not use_bf16,
        # This model uses its declared parameters in the forward path, so unused
        # parameter detection would mainly add DDP synchronization overhead.
        ddp_find_unused_parameters=False,
        # Our collator emits custom multimodal keys such as ``input_features``.
        # Those must survive into the model rather than being auto-pruned.
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = CastFloatInputsTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation", None),
        data_collator=collator,
        tokenizer=processor.tokenizer,
        callbacks=[MakeEveryCheckpointInferableCallback(base_model_path=args_cli.model_path)],
    )

    # Explicit checkpoint path wins. Otherwise optionally auto-discover the most
    # recent local checkpoint when ``--resume 1`` is used.
    resume_from = (args_cli.resume_from or "").strip()
    if not resume_from and args_cli.resume == 1:
        resume_from = find_latest_checkpoint(training_args.output_dir) or ""

    if resume_from:
        if trainer.args.process_index == 0:
            print(f"[resume] resume_from_checkpoint = {resume_from}")
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()


if __name__ == "__main__":
    main()
