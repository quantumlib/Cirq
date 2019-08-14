#!/usr/bin/env bash

# Copyright 2019 The Cirq Developers
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

CIRQ_DEV_VERSION="$(
  set -e

  if ! (return 0 2>/dev/null); then
    echo "Usage:" >&2;
    echo "  source set-dev-version.sh" >&2;
    echo >&2;
    echo "This script sets the environment variable \$CIRQ_DEV_VERSION." >&2;
    exit 1
  fi

  # Get the working directory to the repo root.
  cd "$( dirname "${BASH_SOURCE[0]}" )"
  repo_dir=$(git rev-parse --show-toplevel)
  cd "${repo_dir}"

  PROJECT_NAME=cirq

  ACTUAL_VERSION_LINE=$(cat "${PROJECT_NAME}/_version.py" | tail -n 1)
  ACTUAL_VERSION=`echo $ACTUAL_VERSION_LINE | cut -d'"' -f 2`

  if [[ ${ACTUAL_VERSION_LINE} == *"dev"* ]]; then
    echo "${ACTUAL_VERSION}$(date "+%Y%m%d%H%M%S")"
  fi

  exit 0
)"

if [ $? -ne 0 ]; then
  unset CIRQ_DEV_VERSION
  echo -e '\033[31mFAILED\033[0m'
fi
