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
Transformers-native backend for Qwen3-ASR.

These imports expose the Hugging Face configuration, processor and model class
as a compact backend surface. Keeping the re-export here avoids long import
paths in the inference wrapper while still preserving a layered project
structure.
"""

# 中文学习备注：
# 这个文件本身不做任何模型计算，它只是把 transformers backend 里最核心的三件东西统一导出：
# 1. Config：定义整体结构和关键超参
# 2. Processor：负责 waveform/text -> 模型输入
# 3. Model：负责 audio encoder + thinker decoder 的真正前向与生成
#
# 也就是说，如果你把 `qwen_asr/core/transformers_backend` 当成一个小子系统，
# 这个 `__init__.py` 就是它对外暴露的最小 API 面。

from .configuration_qwen3_asr import Qwen3ASRConfig
from .modeling_qwen3_asr import Qwen3ASRForConditionalGeneration
from .processing_qwen3_asr import Qwen3ASRProcessor
