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


# This script runs checks (e.g. pylint, pytest) against a pull request on
# github or against local code. It is called as follows:
#
# bash continuous-integration/check.sh \
#    [pull-request-number] \
#    [access-token] \
#    [--only=pylint|typecheck|pytest|pytest2|incremental-coverage] \
#    [--verbose]
#
# If no pull request number is given, the script will check files in its
# own directory's github repo. If a pull request number is given, the script
# will fetch the corresponding PR from github and run checks against it.
#
# If an access token is given, to go along with the pull request number, the
# script will attempt to use this token to update the status checks shown on
# github.
#
# In order to only run a subset of the checks, one can pass --only=X arguments.
# It is allowed to specify multiple --only arguments, with the understanding
# that this means to run each of the specified checks but not others.
#
# For debugging purposes, the --verbose argument can be included. This will
# cause significantly more output to be produced (e.g. describing all the
# packages that pip installed).


set -e
own_directory="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd ${own_directory}
cirq_dir=$(git rev-parse --show-toplevel)
cd ${cirq_dir}
export PYTHONPATH=${cirq_dir}:${PYTHONPATH}

python3 ${cirq_dir}/dev_tools/run_checks.py $@
