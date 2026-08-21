# Copyright 2026 The Cirq Developers
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

import collections
import os
import pathlib
import re
import subprocess
import sys


def test_benchmark_names_are_unique() -> None:
    """Ensure every benchmark function has a unique names."""
    # benchmarks are stored in BigQuery under their names rather than full ID-s
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTEST_")}
    benchmarks_dir = pathlib.Path(__file__).parent.absolute()
    cmd: list[str | pathlib.Path] = [
        sys.executable,
        "-m",
        "pytest",
        "--override-ini=python_files=*_perf.py",
        "--collect-only",
        benchmarks_dir,
    ]
    output = subprocess.check_output(cmd, env=env, text=True)
    benchmark_names = [
        match.group(1) for match in re.finditer(r"^ *<Function (test_.*)>$", output, re.MULTILINE)
    ]
    # Verify parsing of benchmark names.  Feel free to adjust the minimum count
    # should we purge benchmarks.
    assert len(benchmark_names) > 100
    duplicate_benchmark_names = [
        name for name, count in collections.Counter(benchmark_names).items() if count > 1
    ]
    assert not duplicate_benchmark_names
