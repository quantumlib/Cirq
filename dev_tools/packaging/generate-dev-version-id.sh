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
# This script prints a new dev version id if the current cirq version
# is a dev release.
#
# If version in Cirq's _version.py file contains 'dev' this
# prints a dev version appended by an id given by the time the
# script was run. This can be used to ensure later releases
# have increasing numerical release numbers.
#
# This script is used by publish-dev-package and can also be
# used to set a bash variable for use by travis ci to deploy
# a new dev version on successful merge.
#
# Example:
#     > echo `generate-dev-version-id.sh`
#     0.6.0.dev20190829135619
################################################################################

set -e

# Get the working directory to the repo root.
cd "$( dirname "${BASH_SOURCE[0]}" )"
repo_dir=$(git rev-parse --show-toplevel)
cd "${repo_dir}"

PROJECT_NAME=cirq-core/cirq

ACTUAL_VERSION_LINE=$(tail -n 1 "${PROJECT_NAME}/_version.py")
ACTUAL_VERSION=$(echo "$ACTUAL_VERSION_LINE" | cut -d'"' -f 2)

if [[ ${ACTUAL_VERSION} == *"dev" ]]; then
  echo "${ACTUAL_VERSION}$(date "+%Y%m%d%H%M%S")"
else
  echo "Version doesn't end in dev: ${ACTUAL_VERSION_LINE}" >&2
  exit 1
fi
