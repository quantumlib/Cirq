# Copyright 2019 The Cirq Developers
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

from __future__ import annotations

import io
import sys


class OutputCapture:
    """A context manager that captures stdout and stderr."""

    def __init__(self):
        self.buffer = io.StringIO()
        self._cache = None

    def __enter__(self):
        self._cache = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = self.buffer, self.buffer

    def __exit__(self, exc_type, exc_value, exc_traceback):
        sys.stdout, sys.stderr = self._cache

    def content(self) -> str:
        return self.buffer.getvalue()
