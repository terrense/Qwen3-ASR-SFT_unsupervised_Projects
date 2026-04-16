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

# 中文学习备注：
# `qwen_asr/__init__.py` 在 Python 包里通常承担“对外门面层”的角色。
# 它不负责真实推理，也不负责音频处理，而是在回答一个更工程化的问题：
# “外部用户 import 这个包时，最应该拿到哪些稳定符号？”
#
# 这里的设计意图是把内部目录结构藏起来，让用户只记住这几个高层入口：
# - `Qwen3ASRModel`：主推理封装类，绝大多数使用场景都从它开始
# - `parse_asr_output`：把模型原始文本输出解析成 `(language, text)`
# - `Qwen3ForcedAligner`：可选的时间戳对齐能力
#
# 因此你可以把这个文件理解成：
# “这个仓库希望别人怎样使用 `qwen_asr` 这个包”的最小公开声明。
from .inference.qwen3_asr import Qwen3ASRModel
from .inference.utils import parse_asr_output

try:
    # Forced aligner 依赖更重，而且可能受语言相关第三方包影响。
    # 这里故意做成“软依赖”：
    # - 如果导入成功，包对外就暴露 `Qwen3ForcedAligner`
    # - 如果导入失败，基础 ASR 仍然能正常使用
    #
    # 这种写法的核心目的，是避免“想用普通 ASR 的用户，却因为没装对齐依赖而整个包都 import 失败”。
    from .inference.qwen3_forced_aligner import Qwen3ForcedAligner
except Exception:
    # 这里不抛错，而是显式置为 `None`，这样上层代码可以用
    # `if Qwen3ForcedAligner is None` 来判断可选能力是否可用。
    Qwen3ForcedAligner = None  # type: ignore[assignment]

# `__all__` 明确声明 `from qwen_asr import *` 时哪些符号属于公开 API。
# 更重要的是，它也在提醒阅读者：上面这几个名字才是“官方承诺稳定”的包级入口。
__all__ = ["Qwen3ASRModel", "Qwen3ForcedAligner", "parse_asr_output"]
