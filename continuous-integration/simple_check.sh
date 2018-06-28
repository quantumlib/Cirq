#!/usr/bin/env bash

# Copyright 2018 The Cirq Developers
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


# This script is a lightweight version of check.sh. This script is not as
# reliable as check.sh because it avoids using tools that require system setup.
# In particular, this script DOES NOT:
#
# 1. Create fresh test environments with updated deps (requires virtualenv).
# 2. Run python 2 compatibility tests (requires the protobuf compiler).
#
# What this script DOES do is run pylint, mypy, pytest, and incremental code
# coverage against your local python 3 dev copy of cirq.


# Get the working directory to the repo root.
own_directory="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${own_directory}
repo_dir=$(git rev-parse --show-toplevel)
cd ${repo_dir}

# Run the checks.
export PYTHONPATH=${repo_dir}:${PYTHONPATH}
python3 ${repo_dir}/dev_tools/run_simple_checks.py $@
result=$?

# Delete coverage files created by pytest during the checks.
find | grep "\.py,cover$" | xargs rm

exit ${result}
