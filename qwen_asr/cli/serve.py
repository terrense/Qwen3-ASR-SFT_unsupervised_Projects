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
"""Thin wrapper around ``vllm serve`` for Qwen3-ASR.

Qwen3-ASR is not one of vLLM's built-in model families, so a plain ``vllm
serve`` command would not know how to instantiate the custom multimodal model
class and processor. This file solves exactly that bootstrapping problem:

1. Register the Hugging Face config/model/processor classes.
2. Register the custom vLLM model implementation.
3. Delegate argument parsing and serving behavior back to upstream vLLM.

The result is operationally simple for users while keeping all Qwen3-ASR-
specific knowledge localized in one small entrypoint.
"""

import sys

from qwen_asr.core.transformers_backend import (
    Qwen3ASRConfig,
    Qwen3ASRForConditionalGeneration,
    Qwen3ASRProcessor,
)
from transformers import AutoConfig, AutoModel, AutoProcessor

# 中文学习备注：
# 这里重复做一遍注册，不是多余，而是因为这个文件本身就可能被当作独立 CLI 入口直接运行。
# 如果不在启动服务前显式注册，自定义 `qwen3_asr` 架构对 vLLM / HF auto 类来说就是“未知模型”。
AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
AutoModel.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration)
AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor)

try:
    from qwen_asr.core.vllm_backend import Qwen3ASRForConditionalGeneration
    from vllm import ModelRegistry
    ModelRegistry.register_model("Qwen3ASRForConditionalGeneration", Qwen3ASRForConditionalGeneration)
except Exception as e:
    raise ImportError(
        "vLLM is not available, to use qwen-asr-serve, please install with: pip install qwen-asr[vllm]"
    ) from e

from vllm.entrypoints.cli.main import main as vllm_main

def main():
    """
    Delegate to the upstream vLLM CLI after injecting the ``serve`` subcommand.

    Mutating ``sys.argv`` is a simple way to reuse vLLM's command-line parser
    without copying its entrypoint implementation.
    """
    # 这招的本质是“借壳”：
    # Qwen3-ASR 只负责在进入 vLLM CLI 之前完成必要注册，
    # 真正的参数解析和服务启动逻辑仍然完全交给上游 vLLM。
    sys.argv.insert(1, "serve")
    vllm_main()


if __name__ == "__main__":
    main()
