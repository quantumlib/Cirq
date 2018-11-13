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
# First argument must be the output directory. Second argument is an optional
# version specifier. If not set, the version from `cirq/_version.py` is used.
# If set, it overwrites `cirq/_version.py`.
#
# Usage:
#     dev_tools/packaging/produce-package.sh output_dir [version]
################################################################################

set -e

if [ -z "${1}" ]; then
  echo -e "\e[31mNo output directory given.\e[0m"
  exit 1
fi
out_dir=$(realpath "${1}")

SPECIFIED_VERSION="${2}"

# Get the working directory to the repo root.
cd "$( dirname "${BASH_SOURCE[0]}" )"
repo_dir=$(git rev-parse --show-toplevel)
cd ${repo_dir}

# Make a clean copy of HEAD, without files ignored by git (but potentially kept by setup.py).
if [ ! -z "$(git status --short)" ]; then
    echo -e "\e[31mWARNING: You have uncommitted changes. They won't be included in the package.\e[0m"
fi
tmp_git_dir=$(mktemp -d "/tmp/produce-package-git.XXXXXXXXXXXXXXXX")
trap "{ rm -rf ${tmp_git_dir}; }" EXIT
cd "${tmp_git_dir}"
git init --quiet
git fetch ${repo_dir} HEAD --quiet --depth=1
git checkout FETCH_HEAD -b work --quiet
if [ ! -z "${SPECIFIED_VERSION}" ]; then
    echo '__version__ = "'"${SPECIFIED_VERSION}"'"' > "${tmp_git_dir}/cirq/_version.py"
fi

# Python 3 wheel.
echo "Producing python 3 package files..."
python3 setup.py -q bdist_wheel -d "${out_dir}"

# Python 2 wheel.
echo "Generating python 2.7 source..."
tmp_py2_dir=$(mktemp -d "/tmp/produce-package-py2.XXXXXXXXXXXXXXXX")
trap "{ rm -rf ${tmp_py2_dir}; }" EXIT
"${tmp_git_dir}/dev_tools/python2.7-generate.sh" "${tmp_py2_dir}/py2" "${tmp_git_dir}"
echo "Producing python 2.7 package files..."
export PYTHONPATH="${tmp_py2_dir}/py2"
cd "${tmp_py2_dir}/py2"
python2 setup.py -q bdist_wheel -d "${out_dir}"

ls "${out_dir}"
