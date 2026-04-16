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
vLLM-native backend for Qwen3-ASR.

The implementation is separated from the Transformers backend because vLLM has
its own execution model, tensor-parallel layers and multimodal registration
hooks. Re-exporting the main class here keeps the higher-level inference code
backend-agnostic.
"""

# 中文学习备注：
# 这个 `__init__.py` 和 transformers backend 的同名文件作用类似，
# 都是在给上层推理封装提供一个更短、更稳定的导入入口。
# 区别在于这里暴露的是“只适用于 vLLM 运行时”的那套模型实现。
#
# 所以当你在 `inference/qwen3_asr.py` 里看到：
# `from qwen_asr.core.vllm_backend import Qwen3ASRForConditionalGeneration`
# 本质上就是通过这里把 vLLM 专用模型类接进上层封装。
from .qwen3_asr import Qwen3ASRForConditionalGeneration
