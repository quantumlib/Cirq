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

# Get the working directory to the repo root.
own_directory="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${own_directory}
repo_dir=$(git rev-parse --show-toplevel)
cd ${repo_dir}

cur_dir=$(pwd)
out_dir="$(pwd)/python2.7-package-tmp"

echo "Generating python 2.7 source (this will take a minute)..."
bash python2.7-generate.sh "${out_dir}" "${cur_dir}"

echo "Producing package..."
export PYTHONPATH=${out_dir}
cd "${out_dir}"
python2 setup.py sdist
cp -r dist/ "${cur_dir}/dist2.7"
cd "${cur_dir}"
rm -rf "${out_dir}"

echo "Done. Output is in 'dist2.7/'."
