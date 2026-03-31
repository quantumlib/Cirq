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
# This script prints a version id if the current Cirq version is a dev release.
#
# If version in Cirq's _version.py file ends in "dev0" this prints
# a dev version appended by the HEAD commit date in the UTC timezone.
# This can be used to ensure later releases have increasing release numbers.
#
# This script is used by the automated "Pre-release cirq to PyPi"
# GitHub action to produce a new dev version after successful merge.
#
# Example:
#     > ./generate-dev-version-id.sh
#     0.6.0.dev20190829135619
################################################################################

set -e

# Get the working directory to the repo root.
thisdir=$(dirname "${BASH_SOURCE[0]:?}")
repo_dir=$(git -C "${thisdir}" rev-parse --show-toplevel)
cd "${repo_dir}"

PROJECT_NAME=cirq-core/cirq

ACTUAL_VERSION_LINE=$(tail -n 1 "${PROJECT_NAME}/_version.py")
ACTUAL_VERSION=$(echo "$ACTUAL_VERSION_LINE" | cut -d'"' -f 2)

if [[ ${ACTUAL_VERSION} != *"dev0" ]]; then
  echo "Version doesn't end in dev0: ${ACTUAL_VERSION_LINE}" >&2
  exit 1
fi

if ! (git diff --cached --quiet && git diff --quiet); then
  echo "There are uncommitted changes in the repository." >&2
  echo "Please commit or clean these up to try again." >&2
  exit 1
fi

TIMESTAMP=$(TZ=UTC git log -1 --date="format-local:%Y%m%d%H%M%S" --pretty="%cd")
echo "${ACTUAL_VERSION%0}${TIMESTAMP}"
