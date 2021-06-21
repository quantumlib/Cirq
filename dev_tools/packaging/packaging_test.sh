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
# This script tests packaging. It only runs if setup.py or requirements (*.txt)
# files are changed on the given branch. It creates the packages for all the
# cirq modules `pip install`s them in a clean virtual environment and then runs
# some simple verificiations on each of the modules, ensuring that they can be
# imported. It then also attempts to push to test pypi, using the
# $TEST_TWINE_USERNAME and $TEST_TWINE_PASSWORD environment variables.
################################################################################

set -e

# Get the working directory to the repo root.
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$(git rev-parse --show-toplevel)"

# Figure out which revision to compare against.
if [ ! -z "$1" ] && [[ $1 != -* ]]; then
    if [ "$(git cat-file -t $1 2> /dev/null)" != "commit" ]; then
        echo -e "\033[31mNo revision '$1'.\033[0m" >&2
        exit 1
    fi
    rev=$1
elif [ "$(git cat-file -t upstream/master 2> /dev/null)" == "commit" ]; then
    rev=upstream/master
elif [ "$(git cat-file -t origin/master 2> /dev/null)" == "commit" ]; then
    rev=origin/master
elif [ "$(git cat-file -t master 2> /dev/null)" == "commit" ]; then
    rev=master
else
    echo -e "\033[31mNo default revision found to compare against. Argument #1 must be what to diff against (e.g. 'origin/master' or 'HEAD~1').\033[0m" >&2
    exit 1
fi
base="$(git merge-base ${rev} HEAD)"
if [ "$(git rev-parse ${rev})" == "${base}" ]; then
    echo -e "Comparing against revision '${rev}'." >&2
else
    echo -e "Comparing against revision '${rev}' (merge base ${base})." >&2
    rev="${base}"
fi

# only run if setup.py or txt changes are observed, to avoid too frequent uploads to test pypi
changed=$(git diff --name-only ${rev} -- \
    | grep -E "^.*(setup.py|.txt)$"
)


# Temporary workspace.
tmp_dir=$(mktemp -d)
trap "{ rm -rf ${tmp_dir}; }" EXIT

# New virtual environment
echo "Working in a fresh virtualenv at ${tmp_dir}/env"
virtualenv --quiet "--python=/usr/bin/python3" "${tmp_dir}/env"

export CIRQ_PRE_RELEASE_VERSION=$(dev_tools/packaging/generate-dev-version-id.sh)
out_dir=${tmp_dir}/dist
dev_tools/packaging/produce-package.sh ${out_dir} $CIRQ_PRE_RELEASE_VERSION

# test installation 
"${tmp_dir}/env/bin/python" -m pip install ${out_dir}/*

echo ===========================
echo Testing that code executes
echo ===========================

"${tmp_dir}/env/bin/python" -c "import cirq; print(cirq.google.Foxtail)"
"${tmp_dir}/env/bin/python" -c "import cirq_google; print(cirq_google.Foxtail)"
"${tmp_dir}/env/bin/python" -c "import cirq; print(cirq.Circuit(cirq.CZ(*cirq.LineQubit.range(2))))"

echo =======================================
echo Testing that all modules are installed
echo =======================================

CIRQ_PACKAGES=$(env PYTHONPATH=. python dev_tools/modules.py list --mode package)
for p in $CIRQ_PACKAGES; do
  echo --- Testing $p -----
  python_test="import $p; print($p); assert '${tmp_dir}' in $p.__file__, 'Package path seems invalid.'"
  env PYTHONPATH='' "${tmp_dir}/env/bin/python" -c "$python_test" && echo -e "\033[32mPASS\033[0m"  || echo -e "\033[31mFAIL\033[0m"
done


echo ============================================
echo Testing that modules can be uploaded to pypi
echo ============================================

twine upload --repository-url=https://test.pypi.org/legacy/ -u="$TEST_TWINE_USERNAME" -p="$TEST_TWINE_PASSWORD" "${out_dir}/*"
