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


# This script fetches a pull request from the quantumlib/cirq repository and
# runs pylint on it. It informs github of the outcome of the linting via the
# 'pylint (manual)' status indicator.
#
#   bash pylint-pull-request.sh [pull_request_number] [access_token]
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
github_context="pylint by maintainer"
source "${own_directory}/load-pull-request-content.sh"

function clean_up_catch () {
  echo "FAILED TO COMPLETE"
  set_status "error" "Script failed."
}
trap clean_up_catch ERR

# Run pylint.
echo
echo "Running pylint..."
py35=${PYTHON35_DIR:-"/usr/bin/python3.5"}
virtualenv --quiet -p "${py35}" "${work_dir}/cirq3.5"
source "${work_dir}/cirq3.5/bin/activate"
pip install --quiet -r requirements.txt
set +e
find cirq | grep "\.py$" | grep -v "_pb2\.py$" | xargs pylint --reports=no --score=no --output-format=colorized --rcfile=continuous-integration/.pylintrc ''
outcome=$?
set -e
deactivate

# Report result.
echo
if [ "${outcome}" -eq 0 ]; then
  echo "Outcome: PASSED"
  set_status "success" "No lint!"
else
  echo -e "Outcome: \e[31mFAILED\e[0m"
  set_status "failure" "Lint present."
fi
