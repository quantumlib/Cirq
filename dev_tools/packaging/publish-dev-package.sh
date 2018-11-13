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

################################################################################
# Produces and uploads dev-version wheels to a pypi package repository. Uploads
# to the test pypi repository unless the --prod switch is added.
#
# Usage:
#     export TEST_TWINE_USERNAME=...
#     export TEST_TWINE_PASSWORD=...
#     export PROD_TWINE_USERNAME=...
#     export PROD_TWINE_PASSWORD=...
#     dev_tools/packaging/publish-dev-package.sh EXPECTED_VERSION [--test|--prod]
#
# The uploaded package can be installed with pip, but if it's the test version
# then the requirements must be installed separately first (because they do not
# all exist on test pypi).
#
# Prod installation:
#
#     pip install cirq==VERSION_YOU_UPLOADED
#
# Python 3 test installation:
#
#     pip install -r requirements.txt
#     pip install --index-url https://test.pypi.org/simple/ cirq==VERSION_YOU_UPLOADED
#
# Python 2 test installation:
#
#     pip install -r dev_tools/python2.7-requirements.txt
#     pip install --index-url https://test.pypi.org/simple/ cirq==VERSION_YOU_UPLOADED
################################################################################

set -e
trap "{ echo -e '\e[31mFAILED\e[0m'; }" ERR

EXPECTED_VERSION=$1
PROD_SWITCH=$2

if [ -z "${EXPECTED_VERSION}" ]; then
    echo -e "\e[31mFirst argument must be the expected version.\e[0m"
    exit 1
fi
if [[ "${EXPECTED_VERSION}" != *dev* ]]; then
  echo -e "\e[31mExpected version must include 'dev'.\e[0m"
  exit 1
fi
ACTUAL_VERSION_LINE=$(cat cirq/_version.py | tail -n 1)
if [ "${ACTUAL_VERSION_LINE}" != '__version__ = "'"${EXPECTED_VERSION}"'"' ]; then
  echo -e "\e[31mExpected version (${EXPECTED_VERSION}) didn't match the one in cirq/_version.py (${ACTUAL_VERSION_LINE}).\e[0m"
  exit 1
fi

if [ -z "${PROD_SWITCH}" ] || [ "${PROD_SWITCH}" = "--test" ]; then
    PYPI_REPOSITORY_FLAG="--repository-url=https://test.pypi.org/legacy/"
    PYPI_REPO_NAME="TEST"
    USERNAME="${TEST_TWINE_USERNAME}"
    PASSWORD="${TEST_TWINE_PASSWORD}"
    if [ -z "${USERNAME}" ]; then
      echo -e "\e[31mTEST_TWINE_USERNAME environment variable must be set.\e[0m"
      exit 1
    fi
    if [ -z "${PASSWORD}" ]; then
      echo -e "\e[31mTEST_TWINE_PASSWORD environment variable must be set.\e[0m"
      exit 1
    fi
elif [ "${PROD_SWITCH}" = "--prod" ]; then
    PYPI_REPOSITORY_FLAG=''
    PYPI_REPO_NAME="PROD"
    USERNAME="${PROD_TWINE_USERNAME}"
    PASSWORD="${PROD_TWINE_PASSWORD}"
    if [ -z "${USERNAME}" ]; then
      echo -e "\e[31mPROD_TWINE_USERNAME environment variable must be set.\e[0m"
      exit 1
    fi
    if [ -z "${PASSWORD}" ]; then
      echo -e "\e[31mPROD_TWINE_PASSWORD environment variable must be set.\e[0m"
      exit 1
    fi
else
    echo -e "\e[31mSecond argument must be empty, '--test' or '--prod'.\e[0m"
    exit 1
fi


UPLOAD_VERSION="${EXPECTED_VERSION}$(date "+%Y%m%d%H%M%S")"
echo -e "Producing package with version \e[33m\e[100m${UPLOAD_VERSION}\e[0m to upload to \e[33m\e[100m${PYPI_REPO_NAME}\e[0m pypi repository"

# Get the working directory to the repo root.
cd "$( dirname "${BASH_SOURCE[0]}" )"
cd "$(git rev-parse --show-toplevel)"

# Temporary workspace.
tmp_package_dir=$(mktemp -d "/tmp/publish-dev-package_package.XXXXXXXXXXXXXXXX")
trap "{ rm -rf ${tmp_package_dir}; }" EXIT

# Produce packages.
dev_tools/packaging/produce-package.sh "${tmp_package_dir}" "${UPLOAD_VERSION}"
twine upload --username="${USERNAME}" --password="${PASSWORD}" ${PYPI_REPOSITORY_FLAG} "${tmp_package_dir}/*"

echo -e "\e[32mUploaded package with version ${UPLOAD_VERSION} to ${PYPI_REPO_NAME} pypi repository\e[0m"
