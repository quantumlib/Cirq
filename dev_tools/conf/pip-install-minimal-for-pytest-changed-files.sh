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

set -e

# Get the working directory to the repo root.
cd "$( dirname "${BASH_SOURCE[0]}" )"
cd "$(git rev-parse --show-toplevel)"

reqs=(
    -r dev_tools/requirements/pytest-minimal.env.txt
    -r dev_tools/requirements/deps/notebook.txt
)

# Install contrib requirements only if needed.
changed=$(git diff --name-only origin/master | grep "cirq/contrib" || true)
[ "${changed}" = "" ] || reqs+=( -r cirq-core/cirq/contrib/requirements.txt )

pip install "${reqs[@]}"
