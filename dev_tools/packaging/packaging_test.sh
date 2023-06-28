#!/bin/bash

# Copyright 2021 The Cirq Developers
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
# This script tests packaging. It creates the packages for all the cirq modules
# `pip install`s them in a clean virtual environment and then runs some simple
# verificiations on each of the modules, ensuring that they can be imported.
################################################################################

set -e

# Temporary workspace.
tmp_dir=$(mktemp -d)
trap '{ rm -rf "${tmp_dir}"; }' EXIT

# New virtual environment
echo "Working in a fresh virtualenv at ${tmp_dir}/env"
python3.9 -m venv "${tmp_dir}/env"

export CIRQ_PRE_RELEASE_VERSION
CIRQ_PRE_RELEASE_VERSION=$(dev_tools/packaging/generate-dev-version-id.sh)
out_dir=${tmp_dir}/dist
dev_tools/packaging/produce-package.sh "${out_dir}" "$CIRQ_PRE_RELEASE_VERSION"

# test installation
"${tmp_dir}/env/bin/python" -m pip install "${out_dir}"/*

echo ===========================
echo Testing that code executes
echo ===========================

"${tmp_dir}/env/bin/python" -c "import cirq; print(cirq.Circuit(cirq.CZ(*cirq.LineQubit.range(2))))"
"${tmp_dir}/env/bin/python" -c "import cirq_google; print(cirq_google.Sycamore)"


echo =======================================
echo Testing that all modules are installed
echo =======================================

CIRQ_PACKAGES=$(env PYTHONPATH=. python dev_tools/modules.py list --mode package)
for p in $CIRQ_PACKAGES; do
  echo "--- Testing $p -----"
  python_test="import $p; print($p); assert '${tmp_dir}' in $p.__file__, 'Package path seems invalid.'"
  env PYTHONPATH='' "${tmp_dir}/env/bin/python" -c "$python_test" && echo -e "\033[32mPASS\033[0m"  || echo -e "\033[31mFAIL\033[0m"
done
