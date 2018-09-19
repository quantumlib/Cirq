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


# This script will keep syncing a PR to master, waiting for its check to
# succeeds, and attempting to merge it; until either the PR is merged, the
# checks fail, or there are merge conflicts. After merging, it deletes the
# PR's branch (unless the branch is used elsewhere) and moves on to the next PR.
# The script does not skip a PR and go on to later PRs when the earlier PR
# enters an impossible-to-merge state; the script just stops.
#
# The commit message used by this script is the title of the PR, and for the
# message body it uses the body of the PR's initial message. So make sure those
# are good if you want good commit messages.
#
# Usage:
#     export CIRQ_GITHUB_PR_ACCESS_TOKEN=[access token for your github account]
#     bash dev_tools/auto_merge.sh [PR#1] [PR#2] ...


# Get the working directory to the repo root.
own_directory="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${own_directory}
repo_dir=$(git rev-parse --show-toplevel)
cd ${repo_dir}

# Do the thing.
export PYTHONPATH=${repo_dir}
python3 ${repo_dir}/dev_tools/auto_merge.py $@
