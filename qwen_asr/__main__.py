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
Console entry for ``python -m qwen_asr``.

The project primarily exposes task-specific executables instead of a full
interactive shell, so this module simply points users toward the intended CLI
commands.
"""

# 中文学习备注：
# 这个文件不是功能入口，而是“命令提示入口”。
# 也就是说，运行 `python -m qwen_asr` 时，它不会真的启动模型服务或 demo，
# 只是告诉用户这个包真正推荐使用的几个 CLI 命令是什么。
def main():
    """Print the supported top-level command line entry points."""
    print(
        "qwen_asr package.\n"
        "Use CLI entrypoints:\n"
        "  - qwen-asr-demo\n"
        "  - qwen-asr-demo-streaming\n"
        "  - qwen-asr-serve\n"
    )

if __name__ == "__main__":
    main()
