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
# This script tests module installations in isolation. It attempts to install
# each cirq module with `pip install ./cirq-core $folder` in a clean virtual
# environment and then runs some simple verificiations on each of the modules
################################################################################

set -e

echo =======================================
echo Testing modules in isolation
echo =======================================

function fail() {
    echo -e "\033[31mFAIL - $1\033[0m"
    exit 1
}

CIRQ_FOLDERS=$(env PYTHONPATH=. python dev_tools/modules.py list --mode folder)
for folder in $CIRQ_FOLDERS; do
  echo --- Testing $p in isolation ---

  # Temporary workspace.
  tmp_dir=$(mktemp -d)
  trap "{ rm -rf ${tmp_dir}; }" EXIT

  # New virtual environment
  echo "Working in a fresh virtualenv at ${tmp_dir}/env"
  virtualenv --quiet "--python=/usr/bin/python3" "${tmp_dir}/env"

  "${tmp_dir}/env/bin/python" -m pip install -r dev_tools/requirements/pytest-minimal-isolated.env.txt

  if [ "$folder" != "cirq-core" ]; then
    echo "-- $folder should not install without cirq-core"
    "${tmp_dir}/env/bin/python" -m pip install ./$folder && fail "$folder should have failed to install without cirq-core!" || echo -e "\033[32mPASS (the above failure was expected!)\033[0m"
  fi
  echo "-- $folder should install successfully with cirq-core"
  "${tmp_dir}/env/bin/python" -m pip install ./cirq-core ./$folder  && echo -e "\033[32mPASS\033[0m"  || fail "'pip install ./cirq-core ./$folder'"
  echo "-- running pytest $folder"
  "${tmp_dir}/env/bin/pytest" ./$folder --ignore ./cirq-core/cirq/contrib

  "${tmp_dir}/env/bin/python" -m pip uninstall cirq-core $folder -y
done