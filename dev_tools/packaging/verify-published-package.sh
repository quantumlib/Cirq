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
# Downloads and tests cirq wheels from the pypi package repository.
# Can verify prod, test, or pre-release versions.
#   --pre: pre-release cirq from prod pypi
#   --test: cirq from test pypi
#   --prod: cirq from prod pypi
#
# CAUTION: when targeting the test pypi repository, this script assumes that the
# local version of cirq has the same dependencies as the remote one (because the
# dependencies must be installed from the non-test pypi repository). If the
# dependencies disagree, the tests can spuriously fail.
#
# Usage:
#     dev_tools/packaging/verify-published-package.sh PACKAGE_VERSION --test|--prod|--pre
################################################################################

set -e
trap "{ echo -e '\033[31mFAILED\033[0m'; }" ERR


PROJECT_NAME=cirq
PROJECT_VERSION=$1
PROD_SWITCH=$2

if [ -z "${PROJECT_VERSION}" ]; then
    echo -e "\033[31mFirst argument must be the package version to test.\033[0m"
    exit 1
fi

if [ "${PROD_SWITCH}" = "--test" ]; then
    PIP_FLAGS="--index-url=https://test.pypi.org/simple/"
    PYPI_REPO_NAME="TEST"
    PYPI_PROJECT_NAME="cirq"
elif [ "${PROD_SWITCH}" = "--prod" ]; then
    PIP_FLAGS=''
    PYPI_REPO_NAME="PROD"
    PYPI_PROJECT_NAME="cirq"
elif [ "${PROD_SWITCH}" = "--pre" ]; then
    PIP_FLAGS='--pre'
    PYPI_REPO_NAME="PROD"
    PYPI_PROJECT_NAME="cirq"
else
    echo -e "\033[31mSecond argument must be '--prod' or '--test' or '--pre'.\033[0m"
    exit 1
fi

# Find the repo root.
cd "$( dirname "${BASH_SOURCE[0]}" )"
REPO_ROOT="$(git rev-parse --show-toplevel)"

# Temporary workspace.
tmp_dir=$(mktemp -d "/tmp/verify-published-package.XXXXXXXXXXXXXXXX")
cd "${tmp_dir}"
trap '{ rm -rf "${tmp_dir}"; }' EXIT

# Test installation from published package
PYTHON_VERSION=python3

# Prepare.
CONTRIB_DEPS_FILE="${REPO_ROOT}/cirq-core/cirq/contrib/requirements.txt"
DEV_DEPS_FILE="${REPO_ROOT}/dev_tools/requirements/deps/dev-tools.txt"

echo -e "\n\033[32m${PYTHON_VERSION}\033[0m"
echo "Working in a fresh virtualenv at ${tmp_dir}/${PYTHON_VERSION}"
virtualenv --quiet "--python=/usr/bin/${PYTHON_VERSION}" "${tmp_dir}/${PYTHON_VERSION}"

echo Installing "${PYPI_PROJECT_NAME}==${PROJECT_VERSION} from ${PYPI_REPO_NAME} pypi"
"${tmp_dir}/${PYTHON_VERSION}/bin/pip" install --quiet ${PIP_FLAGS} "${PYPI_PROJECT_NAME}==${PROJECT_VERSION}" --extra-index-url https://pypi.python.org/simple

# Check that code runs without dev deps.
echo Checking that code executes
"${tmp_dir}/${PYTHON_VERSION}/bin/python" -c "import cirq_google; print(cirq_google.Sycamore)"
"${tmp_dir}/${PYTHON_VERSION}/bin/python" -c "import cirq; print(cirq.Circuit(cirq.CZ(*cirq.LineQubit.range(2))))"

# Install pytest + dev deps.
"${tmp_dir}/${PYTHON_VERSION}/bin/pip" install -r "${DEV_DEPS_FILE}"

# Run tests.
PY_VER=$(ls "${tmp_dir}/${PYTHON_VERSION}/lib")
echo Running cirq tests
cirq_dir="${tmp_dir}/${PYTHON_VERSION}/lib/${PY_VER}/site-packages/${PROJECT_NAME}"
"${tmp_dir}/${PYTHON_VERSION}/bin/pytest" --quiet --disable-pytest-warnings --ignore="${cirq_dir}/contrib" "${cirq_dir}"

echo "Installing contrib dependencies"
"${tmp_dir}/${PYTHON_VERSION}/bin/pip" install --quiet -r "${CONTRIB_DEPS_FILE}"

echo "Running contrib tests"
"${tmp_dir}/${PYTHON_VERSION}/bin/pytest" --quiet --disable-pytest-warnings "${cirq_dir}/contrib"

echo
echo -e '\033[32mVERIFIED\033[0m'
