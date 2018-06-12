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
own_directory="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
repo_root="$( cd "$own_directory/.." && pwd )"

if [ -d "$repo_root/docs/_build" ] ; then
  echo "'$repo_root/docs/_build' directory still exists." >&2
  echo "Remove the _build directory before continuing." >&2
  exit 1
fi

if [ -d "$repo_root/docs/generated" ] ; then
  echo "'$repo_root/docs/generated' directory still exists." >&2
  echo "Remove the generated directory before continuing." >&2
  exit 1
fi

cd "$repo_root/docs"
pip install -r dev-requirements.txt
make html

