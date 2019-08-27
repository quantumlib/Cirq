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

################################################################################
# If cirq's version is a dev release, this script can be used to set
# the CIRQ_DEV_VERSION variable to a string that is the dev version
# plus an increasing date string.
#
# This is used by travis during deployment and by published-dev-package.sh.
#
# The variable can be set using
#
#     export CIRQ_DEV_VERSION=`dev_tools/packaging/set-dev-version.sh`
#
# Which, for example sets $CIRQ_DEV_VERSION to 0.6.0.dev20190813193556.
################################################################################

set -e

# Get the working directory to the repo root.
cd "$( dirname "${BASH_SOURCE[0]}" )"
repo_dir=$(git rev-parse --show-toplevel)
cd "${repo_dir}"

PROJECT_NAME=cirq

ACTUAL_VERSION_LINE=$(cat "${PROJECT_NAME}/_version.py" | tail -n 1)
ACTUAL_VERSION=`echo $ACTUAL_VERSION_LINE | cut -d'"' -f 2`

if [[ ${ACTUAL_VERSION} == *"dev" ]]; then
  echo "${ACTUAL_VERSION}$(date "+%Y%m%d%H%M%S")"
else
  echo "Version doesn't end in dev: ${ACTUAL_VERSION_LINE}" >&2
  exit 1
fi

exit 0
