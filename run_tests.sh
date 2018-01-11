#!/usr/bin/env bash

# Copyright 2018 Google LLC
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

# This script runs tests for both python 2.7 and python 3.5. To invoke run
#
#   run_tests [commit_id] [access_token]
#
# If commit_id and access_token are specified the status on github will be updated.
#
# The shell requires that virtualenv has been installed.

set -e
base_dir=$(mktemp -d '/tmp/test-cirq-XXXXX')
py35=${PYTHON35_DIR:-"/usr/bin/python3.5"}
py27=${PYTHON27_DIR:-"/usr/bin/python2.7"}

function clean_up () {
  rm -rf "${base_dir}"
}
trap clean_up EXIT

# Python 3.5 tests.
virtualenv -p "${py35}" "${base_dir}/cirq3.5"
source "${base_dir}/cirq3.5/bin/activate"
pip install -r requirements.txt
pytest cirq
deactivate

# Python 2.7 tests.
virtualenv -p "${py27}" "${base_dir}/cirq2.7"
source "${base_dir}/cirq2.7/bin/activate"
pip install -r python2.7-requirements.txt
pip install 3to2
./python2.7-generate.sh
mv python2.7-output $base_dir
pytest "${base_dir}/python2.7-output/cirq"
deactivate

echo "All tests passed."

if [ -n "$1" ] && [ -n "$2" ]; then
  echo "Updating github..."
  curl -d '{"state":"success","target_url": "https://example.com",' \
    '"description": "Tests passed! \(manual\)", "context": "pytest"}' \
    -X POST https://api.github.com/repos/quantumlib/cirq/statuses/"$1"?access_token="$2"
  echo "Successfully updated github tested status."
else
  echo "No commit id and access token specified, not updating github"
fi
