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
own_directory="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${own_directory}
repo_dir=$(git rev-parse --show-toplevel)
cd ${repo_dir}

# Fail if package files already exist.
mkdir dist

echo "Producing python 3 package files..."
python3 setup.py -q sdist bdist_wheel

echo "Generating python 2.7 source..."
cur_dir=$(pwd)
tmp_py2_dir="$(pwd)/python2.7-package-tmp"
bash dev_tools/python2.7-generate.sh "${tmp_py2_dir}" "${cur_dir}"

echo "Producing python 2.7 package files..."
export PYTHONPATH=${tmp_py2_dir}
cd "${tmp_py2_dir}"
python2 setup.py -q sdist bdist_wheel
cp dist/* "${cur_dir}/dist/"
cd "${cur_dir}"
rm -rf "${tmp_py2_dir}"

ls dist/
echo "Done. Output is in 'dist/'."
