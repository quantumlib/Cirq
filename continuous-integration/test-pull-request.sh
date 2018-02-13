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


pull_id=${1}
access_token=${2:-${CIRQ_GITHUB_ACCESS_TOKEN}}

if [ -z "${pull_id}" ]; then
  echo "No pull request id given, using local files."
else
  if [ -z "${access_token}" ]; then
    echo "No access token given. Won't update github status."
  fi
fi

function set_status () {
  state=$1
  desc=$2
  local update_status="success"
  if [ -n "${pull_id}" ] && [ -n "${access_token}" ]; then
    json='{
            "state": "'${state}'",
            "description": "'${desc}'",
            "context": "pytest (manual)"
        }'
    url="https://api.github.com/repos/quantumlib/cirq/statuses/${commit_id}?access_token=${access_token}"
    curl_result=$(curl --silent -d "${json}" -X POST "${url}")
    if [[ $curl_result = *"${state}"* ]]; then
      update_status="success"
    else
      update_status=$curl_result
    fi
  fi
  echo $update_status
}

work_dir="$(mktemp -d '/tmp/test-cirq-XXXXXXXX')"
set -e
function clean_up_finally () {
  rm -rf "${work_dir}"
}
function clean_up_catch () {
  echo "FAILED TO COMPLETE"
  set_status "error" "Script failed."
}
trap clean_up_finally EXIT ERR
trap clean_up_catch ERR

# Get content to test.
if [ -z "${pull_id}" ]; then
  origin=$(git rev-parse --show-toplevel)
  cp -r "${origin}"/* "${work_dir}"
  cd "${work_dir}"
else
  echo "Fetching pull request #${pull_id}..."
  branch="pull/${pull_id}/head"
  origin="git@github.com:quantumlib/cirq.git"
  cd "${work_dir}"
  git init --quiet
  git fetch git@github.com:quantumlib/cirq.git pull/${pull_id}/head --depth=1 --quiet
  commit_id="$(git rev-parse FETCH_HEAD)"
  git checkout "${commit_id}" --quiet
  result=$(set_status "pending" "Running...")
  if [ "${result}" != "success" ]; then
    echo "FAILED: Could not update status for pull request."
    echo "Curl update status returned: #${result}"
    exit 1
  fi
fi

# Run python 3.5 tests.
echo
echo "Running python 3.5 tests..."
py35=${PYTHON35_DIR:-"/usr/bin/python3.5"}
virtualenv --quiet -p "${py35}" "${work_dir}/cirq3.5"
source "${work_dir}/cirq3.5/bin/activate"
pip install --quiet -r requirements.txt
set +e
pytest --quiet cirq
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
bash python2.7-generate.sh "${work_dir}/python2.7-output"
set +e
pytest --quiet "${work_dir}/python2.7-output/cirq"
outcome_v27=$?
set -e
deactivate

# Report result.
echo
if [ "${outcome_v35}" -eq 0 ] && [ "${outcome_v27}" -eq 0 ]; then
  result=$(set_status "success" "Tests passed!")
  if [ "${result}" = "success" ]; then
    echo "Outcome: SUCCESS"
  else
    echo "Outcome: FAILED"
    echo "Tests passed, but could not update status for pull request."
    echo "Curl update status returned: #${result}"
  fi
else
  result=$(set_status "failure" "Tests failed.")
  if [ "${result}" = "success" ]; then
    echo "Outcome: FAILED"
    echo "Tests did not pass."
  else
    echo "Outcome: FAILED"
    echo "Tests did not pass and could not update status on github"
    echo "Curl update status returned: #${result}"
  fi
fi

