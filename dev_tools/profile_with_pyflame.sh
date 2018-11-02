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
# Profiles a python file, producing output that can be viewed in Chrome's
# JavaScript Profiler.
#
# The profile output is printed to stdout. Any output from the python file will
# be redirected to stderr to avoid cross-contamination.
#
# Usage:
#     dev_tools/profile_with_pyflame.sh FILE_TO_PROFILE.py [custom-flags-for-pyflame] > OUTPUT_FILE.cpuprofile
#
# Requires:
#     pyflame
#         https://github.com/uber/pyflame
#         https://pyflame.readthedocs.io/en/latest/index.html
#     chrome
#         to view the output
#
# Viewing output:
#    Open Chrome to about:blank
#    Open Chrome's console (CTRL+SHIFT+J)
#    DON'T go to the performance tab. It silently fails at opening the file.
#    At the top right of the conolse, hit the 'three dots' next to the X.
#    Select 'More Tools -> Javascript Profiler'.
#    Hit the 'Load' button and select the file this script generated.
################################################################################

# Check dependencies.
if which pyflame > /dev/null; then
    true
else
    echo -e "\e[31mpyflame not found. Follow instructions at https://pyflame.readthedocs.io/en/latest/installation.html and make sure the pyflame directory is in your path.\e[0m" >&2
    exit 1
fi

if which flame-chart-json > /dev/null; then
    true
else
    echo -e "\e[31mflame-chart-json not found. Make sure the pyflame utils directory is in your path.\e[0m" >&2
    exit 1
fi


if [ -z "$1" ]; then
    echo -e "\e[31mSpecify a python file to invoke.\nUSAGE\n    dev_tools/profile_with_pyflame.sh FILE_TO_PROFILE.py [custom-flags-for-pyflame] > OUTPUT_FILE.cpuprofile\e[0m" >&2
    exit 1
fi
if [ -e "$1" ]; then
    true
else
    echo -e "\e[31mExpected a python file but got '${1}', which doesn't exist.\nUSAGE\n    dev_tools/profile_with_pyflame.sh FILE_TO_PROFILE.py [custom-flags-for-pyflame] > OUTPUT_FILE.cpuprofile\e[0m" >&2
    exit 1
fi

# Find repo root.
cd "$( dirname "${BASH_SOURCE[0]}" )"
repo_dir=$(git rev-parse --show-toplevel)
PYTHONPATH="${PYTHONPATH}":"${repo_dir}"
cd - > /dev/null

tmp_out=$(mktemp "/tmp/pyflame.XXXXXXXXXXXXXXXX")
trap "{ rm ${tmp_out}; }" EXIT

pyflame -x --flamechart "${@:2}" -o "${tmp_out}" -t python "$1" 1>&2
cat "${tmp_out}" | flame-chart-json | python "${repo_dir}/dev_tools/clean_up_pyflame_json_for_chrome.py"
