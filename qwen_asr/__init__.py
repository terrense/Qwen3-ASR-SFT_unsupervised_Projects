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
Public package surface for Qwen3-ASR.

This module intentionally re-exports the high-level inference wrappers so that
end users can write concise imports such as ``from qwen_asr import
Qwen3ASRModel`` without needing to know the internal directory layout.

From a Python packaging perspective this file acts as the stable API boundary:
internal submodules may evolve, but callers are expected to depend on the
symbols imported here.
"""

from .inference.qwen3_asr import Qwen3ASRModel
from .inference.qwen3_forced_aligner import Qwen3ForcedAligner

from .inference.utils import parse_asr_output

__all__ = ["__version__"]
