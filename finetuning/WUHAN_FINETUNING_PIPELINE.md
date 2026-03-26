## Wuhan Dialect Adaptation Pipeline

This workflow adds three **new** files under `finetuning/` and does not modify the stock `qwen3_asr_sft.py`.

### 1) Prepare labeled data

Input can be `csv`, `tsv`, `json`, or `jsonl` with at least:

- `audio`
- `text`
- optional `prompt`

Example:

```bash
python finetuning/prepare_asr_jsonl.py \
  --input_manifest ./data/wuhan_labeled.csv \
  --audio_root ./data/audio \
  --language Chinese \
  --output_train ./data/wuhan_train.jsonl \
  --output_eval ./data/wuhan_dev.jsonl \
  --eval_ratio 0.1
```

If `text` does not already contain `<asr_text>`, the script converts it to:

```text
language Chinese<asr_text>...
```

For Wuhan dialect ASR, that is usually the safest target format because the public inference API only exposes coarse language forcing such as `Chinese`, not a dedicated `Wuhan` language flag.

### 2) Generate pseudo labels from unlabeled audio

You can pseudo-label either a manifest or a whole directory:

```bash
python finetuning/generate_pseudo_labels.py \
  --model_path Qwen/Qwen3-ASR-1.7B \
  --audio_dir ./data/wuhan_unlabeled \
  --output_file ./data/wuhan_pseudo.jsonl \
  --decode_language Chinese \
  --train_language Chinese \
  --batch_size 4 \
  --loss_weight 0.3
```

Notes:

- `--decode_language Chinese` forces decoding as Chinese.
- `--train_language Chinese` stores pseudo labels in the same training target format as supervised data.
- `--loss_weight 0.3` writes a weaker default sample weight into pseudo-labeled rows.

### 3) Run mixed supervised + pseudo training

Use the new semi-supervised training script:

```bash
python finetuning/qwen3_asr_sft_semisup.py \
  --model_path Qwen/Qwen3-ASR-1.7B \
  --supervised_train_file ./data/wuhan_train.jsonl \
  --pseudo_train_file ./data/wuhan_pseudo.jsonl \
  --eval_file ./data/wuhan_dev.jsonl \
  --output_dir ./outputs/qwen3_asr_wuhan_semisup \
  --pseudo_loss_weight 0.3 \
  --batch_size 4 \
  --grad_acc 8 \
  --lr 1e-5 \
  --epochs 2
```

### How weighting works

- Supervised samples default to `loss_weight=1.0`.
- Pseudo samples default to `loss_weight=0.3`.
- If a JSONL row already has a `loss_weight` field, that row-level value is preserved.

The semi-supervised trainer computes token-level cross-entropy, averages it per sample, and then applies the sample weight. This is intentionally conservative: noisy pseudo labels should influence training, but not dominate it.

### Suggested Wuhan dialect recipe

1. Start with supervised-only training to get a clean baseline.
2. Pseudo-label a larger unlabeled Wuhan corpus.
3. Mix the two with `pseudo_loss_weight` in the `0.2` to `0.5` range.
4. Keep the target transcript style consistent. If your labels are normalized written Chinese, keep pseudo labels normalized the same way.
5. Evaluate on a held-out Wuhan speaker split with CER, not only training loss.

### File list

- `finetuning/asr_data_utils.py`
- `finetuning/prepare_asr_jsonl.py`
- `finetuning/generate_pseudo_labels.py`
- `finetuning/qwen3_asr_sft_semisup.py`
