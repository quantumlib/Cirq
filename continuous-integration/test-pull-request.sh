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


# This script fetches a pull request from the quantumlib/cirq repository and
# runs tests for both python 2.7 and python 3.5. It informs github of the
# outcome of the testing via the 'pytest (manual)' status indicator.
#
#   bash test-pull-request.sh [pull_request_number] [access_token]
#
# The pull request number argument is optional, and determines which PR is
# fetched from the cirq repo to test. If no pull request number is given, the
# tests run against local files; the current repo state.
#
# The access_token argument is optional. If it is set to a github access token
# with permission to update status values on the cirq repo, then the status of
# the tests will be reported to github and shown within the PR. If not set, the
# tests merely run locally. If this arugment is not set, the script will fall
# back to the CIRQ_GITHUB_ACCESS_TOKEN environment variable (if present).
#
# Requires that virtualenv has been installed.


set -e
own_directory="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
github_context="pytest by maintainer"
source "${own_directory}/load-pull-request-content.sh"

function clean_up_catch () {
  echo "FAILED TO COMPLETE"
  set_status "error" "Script failed."
}
trap clean_up_catch ERR

# Run python 3.5 tests.
echo
echo "Running python 3.5 tests..."
py35=${PYTHON35_DIR:-"/usr/bin/python3.5"}
virtualenv --quiet -p "${py35}" "${work_dir}/cirq3.5"
source "${work_dir}/cirq3.5/bin/activate"
pip install --quiet -r requirements.txt
set +e
pytest --quiet "${work_dir}"
outcome_v35=$?
set -e
deactivate

# Run python 2.7 tests.
echo
echo "Running python 2.7 tests..."
py27=${PYTHON27_DIR:-"/usr/bin/python2.7"}
virtualenv --quiet -p "${py27}" "${work_dir}/cirq2.7"
source "${work_dir}/cirq2.7/bin/activate"
pip install --quiet -r python2.7-requirements.txt
pip install --quiet 3to2
bash python2.7-generate.sh "${work_dir}/python2.7-output" "${work_dir}"
set +e
pytest --quiet "${work_dir}/python2.7-output/cirq"
outcome_v27=$?
set -e
deactivate

# Report result.
echo
if [ "${outcome_v35}" -eq 0 ] && [ "${outcome_v27}" -eq 0 ]; then
  echo "Outcome: PASSED"
  set_status "success" "Tests passed!"
else
  echo -e "Outcome: \e[31mFAILED\e[0m"
  set_status "failure" "Tests failed."
fi
