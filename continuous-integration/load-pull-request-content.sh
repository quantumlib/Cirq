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


# This script-helper fetches a pull request from the quantumlib/cirq repository
# and defines a set_status function for informing github of the outcome of
# local checking.
#
# To use this script, `source` it within another script:
#
#   source test-pull-request.sh
#
# It is expected that the other script has [pull_request_number] [access_token]
# arguments, and that the other script has set github_context to the desired
# status name.
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


set -e

pull_id=${1}
access_token=${2:-${CIRQ_GITHUB_ACCESS_TOKEN}}

if [ -z "${pull_id}" ]; then
  echo -e "\e[31mNo pull request id given, using local files.\e[0m"
else
  if [ -z "${access_token}" ]; then
    echo -e "\e[31mNo access token given. Won't update github status.\e[0m"
  fi
fi

function set_status () {
  state=$1
  desc=$2
  if [ -n "${pull_id}" ] && [ -n "${access_token}" ]; then
    json='{
            "state": "'${state}'",
            "description": "'${desc}'",
            "context": "'${github_context}'"
        }'
    url="https://api.github.com/repos/quantumlib/cirq/statuses/${commit_id}?access_token=${access_token}"
    curl_result=$(curl --silent -d "${json}" -X POST "${url}")
    if [[ ${curl_result} != *"${state}"* ]]; then
      echo "FAILED: Could not update status for pull request."
      echo "Curl update status returned: #${curl_result}"
      exit 1
    fi
  fi
}

work_dir="$(mktemp -d '/tmp/test-cirq-XXXXXXXX')"
function clean_up_finally () {
  rm -rf "${work_dir}"
}
trap clean_up_finally EXIT ERR

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
  set_status "pending" "Running..."
fi
