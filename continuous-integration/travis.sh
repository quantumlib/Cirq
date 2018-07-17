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


# This is the script invoked by travis-ci when running builds.

set -e

if [ "${1}" = "normal" ]; then
    export PYTHONPATH=$(pwd)
    python dev_tools/run_simple_checks.py

elif [ "${1}" = "convert-and-test" ]; then
    cur_dir=$(pwd)
    out_dir="$(pwd)/python2.7-output"

    # Convert code from python3 to python2.7.
    echo "Running 3to2..."
    bash python2.7-generate.sh "${out_dir}" "${cur_dir}"
    echo "Finished conversion."

    # Run tests against converted code.
    export PYTHONPATH=${out_dir}
    pytest ${out_dir}

else
    echo "Unrecognized mode: ${1}"
    exit 1
fi
