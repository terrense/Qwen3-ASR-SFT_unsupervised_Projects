"""Custom launcher for the built-in streaming demo with safer vLLM settings.

The upstream ``qwen_asr.cli.demo_streaming`` script is convenient, but it does
not expose all vLLM memory knobs on the command line. On consumer GPUs with
limited VRAM, Qwen3-ASR may fail to start because vLLM tries to reserve KV cache
for the model's full declared maximum context length.

This launcher reuses the same Flask app and browser UI, but lets us pass
``max_model_len`` and other practical runtime parameters so streaming can run on
smaller cards such as 12 GB Windows GPUs.
"""

from __future__ import annotations

import argparse

from qwen_asr import Qwen3ASRModel
from qwen_asr.cli import demo_streaming as streaming_demo


def parse_args():
    p = argparse.ArgumentParser("Qwen3-ASR Streaming Launcher (custom)")
    p.add_argument("--asr-model-path", default="Qwen/Qwen3-ASR-0.6B", help="Model name or local path")
    p.add_argument("--host", default="0.0.0.0", help="Bind host")
    p.add_argument("--port", type=int, default=8000, help="Bind port")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--unfixed-chunk-num", type=int, default=4)
    p.add_argument("--unfixed-token-num", type=int, default=5)
    p.add_argument("--chunk-size-sec", type=float, default=1.0)
    return p.parse_args()


def main():
    args = parse_args()

    streaming_demo.UNFIXED_CHUNK_NUM = args.unfixed_chunk_num
    streaming_demo.UNFIXED_TOKEN_NUM = args.unfixed_token_num
    streaming_demo.CHUNK_SIZE_SEC = args.chunk_size_sec

    llm_kwargs = dict(
        model=args.asr_model_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_new_tokens=args.max_new_tokens,
    )
    if args.enforce_eager:
        llm_kwargs["enforce_eager"] = True

    streaming_demo.asr = Qwen3ASRModel.LLM(**llm_kwargs)
    print(
        "[streaming] model loaded "
        f"(gpu_memory_utilization={args.gpu_memory_utilization}, "
        f"max_model_len={args.max_model_len}, enforce_eager={args.enforce_eager})"
    )
    streaming_demo.app.run(
        host=args.host,
        port=args.port,
        debug=False,
        use_reloader=False,
        threaded=True,
    )


if __name__ == "__main__":
    main()
