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
# Produces and uploads wheels to the pypi package repository.
#
# Usage:
#     export TEST_TWINE_USERNAME=...
#     export TEST_TWINE_PASSWORD=...
#     dev_tools/public-to-test-pypi.sh output_dir
################################################################################

set -e

if [ -z "${TEST_TWINE_USERNAME}" ]; then
  echo -e "\e[31mTEST_TWINE_USERNAME environment variable must be set.\e[0m"
  exit 1
fi
if [ -z "${TEST_TWINE_PASSWORD}" ]; then
  echo -e "\e[31mTEST_TWINE_PASSWORD environment variable must be set.\e[0m"
  exit 1
fi

# Get the working directory to the repo root.
cd "$( dirname "${BASH_SOURCE[0]}" )"
cd "$(git rev-parse --show-toplevel)"

# Temporary workspace.
tmp_dir=$(mktemp -d "/tmp/publish-to-test-pypi.XXXXXXXXXXXXXXXX")
trap "{ rm -rf ${tmp_dir}; }" EXIT
trap "exit" INT

dev_tools/produce-package.sh "${tmp_dir}"
twine upload --username=${TEST_TWINE_USERNAME} --password=${TEST_TWINE_PASSWORD} --repository-url=https://test.pypi.org/legacy/ "${tmp_dir}/*"
