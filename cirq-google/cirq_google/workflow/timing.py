# Copyright 2022 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for timing execution."""

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import cirq_google as cg


class TimeIntoRuntimeInfo:
    """A context manager that appends timing information into a cg.RuntimeInfo.

    Timings are reported in fractional seconds as reported by `time.monotonic()`.

    Args:
        runtime_info: The runtime information object whose `.timing` dictionary will be updated.
        name: A string key name to use in the dictionary.
    """

    def __init__(self, runtime_info: 'cg.RuntimeInfo', name: str):
        self.runtime_info = runtime_info
        self.name = name
        self.start = None

    def __enter__(self):
        self.start = time.monotonic()
        return self

    def __exit__(self, *args):
        end = time.monotonic()
        interval = end - self.start
        self.runtime_info.timings_s[self.name] = interval
