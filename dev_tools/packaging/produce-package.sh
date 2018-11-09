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

################################################################################
# Produces wheels that can be uploaded to the pypi package repository.
#
# Usage:
#     dev_tools/packaging/produce-package.sh output_dir
################################################################################

set -e

if [ -z "${1}" ]; then
  echo -e "\e[31mNo output directory given.\e[0m"
  exit 1
fi
out_dir=$(realpath "${1}")

# Get the working directory to the repo root.
cd "$( dirname "${BASH_SOURCE[0]}" )"
repo_dir=$(git rev-parse --show-toplevel)
cd ${repo_dir}

echo "Producing python 3 package files..."
python3 setup.py -q bdist_wheel -d "${out_dir}"

echo "Generating python 2.7 source..."
tmp_py2_dir=$(mktemp -d "/tmp/produce-package-py2.XXXXXXXXXXXXXXXX")
trap "{ rm -rf ${tmp_py2_dir}; }" EXIT
rmdir "${tmp_py2_dir}"
bash dev_tools/python2.7-generate.sh "${tmp_py2_dir}" "${repo_dir}"

echo "Producing python 2.7 package files..."
export PYTHONPATH=${tmp_py2_dir}
cd "${tmp_py2_dir}"
python2 setup.py -q bdist_wheel -d "${out_dir}"

ls "${out_dir}"
