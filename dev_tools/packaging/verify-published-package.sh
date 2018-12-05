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
# Downloads and tests cirq wheels from the pypi package repository. Uses the
# prod pypi repository unless the --test switch is added.
#
# CAUTION: when targeting the test pypi repository, this script assumes that the
# local version of cirq has the same dependencies as the remote one (because the
# dependencies must be installed from the non-test pypi repository). If the
# dependencies disagree, the tests can spuriously fail.
#
# Usage:
#     dev_tools/packaging/verify-published-package.sh PACKAGE_VERSION [--test|--prod]
################################################################################

set -e
trap "{ echo -e '\e[31mFAILED\e[0m'; }" ERR


PROJECT_NAME=cirq
PROJECT_VERSION=$1
PROD_SWITCH=$2

if [ -z "${PROJECT_VERSION}" ]; then
    echo -e "\e[31mFirst argument must be the package version to test.\e[0m"
    exit 1
fi

if [ "${PROD_SWITCH}" = "--test" ]; then
    PYPI_REPOSITORY_FLAG="--index-url=https://test.pypi.org/simple/"
    PYPI_REPO_NAME="TEST"
elif [ -z "${PROD_SWITCH}" ] || [ "${PROD_SWITCH}" = "--prod" ]; then
    PYPI_REPOSITORY_FLAG=''
    PYPI_REPO_NAME="PROD"
else
    echo -e "\e[31mSecond argument must be empty, '--prod' or '--test'.\e[0m"
    exit 1
fi

# Find the repo root.
cd "$( dirname "${BASH_SOURCE[0]}" )"
REPO_ROOT="$(git rev-parse --show-toplevel)"

# Temporary workspace.
tmp_dir=$(mktemp -d "/tmp/verify-published-package.XXXXXXXXXXXXXXXX")
cd "${tmp_dir}"
trap "{ rm -rf ${tmp_dir}; }" EXIT

# Test both the python 2 and python 3 versions.
for PYTHON_VERSION in python2 python3; do
    # Prepare.
    if [ "${PYTHON_VERSION}" = "python2" ]; then
        RUNTIME_DEPS_FILE="${REPO_ROOT}/dev_tools/python2.7-requirements.txt"
    else
        RUNTIME_DEPS_FILE="${REPO_ROOT}/requirements.txt"
    fi
    CONTRIB_DEPS_FILE="${REPO_ROOT}/cirq/contrib/contrib-requirements.txt"

    echo -e "\n\e[32m${PYTHON_VERSION}\e[0m"
    echo "Working in a fresh virtualenv at ${tmp_dir}/${PYTHON_VERSION}"
    virtualenv --quiet "--python=/usr/bin/${PYTHON_VERSION}" "${tmp_dir}/${PYTHON_VERSION}"

    # Install package.
    if [ "${PYPI_REPO_NAME}" == "TEST" ]; then
        echo "Pre-installing dependencies since they don't all exist in TEST pypi"
        "${tmp_dir}/${PYTHON_VERSION}/bin/pip" install --quiet -r "${RUNTIME_DEPS_FILE}"
    fi
    echo Installing "${PROJECT_NAME}==${PROJECT_VERSION} from ${PYPI_REPO_NAME} pypi"
    "${tmp_dir}/${PYTHON_VERSION}/bin/pip" install --quiet ${PYPI_REPOSITORY_FLAG} "${PROJECT_NAME}==${PROJECT_VERSION}"

    # Check that code runs without dev deps.
    echo Checking that code executes
    "${tmp_dir}/${PYTHON_VERSION}/bin/python" -c "import cirq; print(cirq.google.Foxtail)"
    "${tmp_dir}/${PYTHON_VERSION}/bin/python" -c "import cirq; print(cirq.Circuit.from_ops(cirq.CZ(*cirq.LineQubit.range(2))))"

    # Run tests.
    echo Installing pytest requirements
    if [ "${PYTHON_VERSION}" = "python2" ]; then
        "${tmp_dir}/${PYTHON_VERSION}/bin/pip" install --quiet pytest mock
    else
        "${tmp_dir}/${PYTHON_VERSION}/bin/pip" install --quiet pytest
    fi
    PY_VER=$(ls "${tmp_dir}/${PYTHON_VERSION}/lib")
    echo Running cirq tests
    cirq_dir="${tmp_dir}/${PYTHON_VERSION}/lib/${PY_VER}/site-packages/${PROJECT_NAME}"
    "${tmp_dir}/${PYTHON_VERSION}/bin/pytest" --quiet --disable-pytest-warnings --ignore="${cirq_dir}/contrib" "${cirq_dir}"

    echo "Installing contrib dependencies"
    "${tmp_dir}/${PYTHON_VERSION}/bin/pip" install --quiet -r "${CONTRIB_DEPS_FILE}"

    echo "Running contrib tests"
    "${tmp_dir}/${PYTHON_VERSION}/bin/pytest" --quiet --disable-pytest-warnings "${cirq_dir}/contrib"
done

echo
echo -e '\e[32mVERIFIED\e[0m'
