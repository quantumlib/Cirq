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

########################################################################################
# This script will continuously watch a github repository for PRs labelled with the
# 'automerge' label. When it sees such a PR it will mark it by labelling it with
# the 'front_of_queue_automerge' label. While there is an 'front_of_queue_automerge'
# labelled PR, the script will not label any other PRs with 'front_of_queue_automerge'.
#
# While there is a 'front_of_queue_automerge' labelled PR, the script will sync that PR
# with master, wait for status checks to succeed, and attempt to merge it into master.
# If the PR goes out of date due to an intervening merge, the process will start over.
# This will continue until either the PR is merged or there is a problem that must be
# addressed by a human. After merging, the PR will be deleted unless it belongs to a
# fork.
#
# The commit message used when merging a PR is the title of the PR and then, for details,
# the body of the PR's initial message/comment. Users/admins should edit the title and
# initial comment to appropriately describe the PR.
#
# This script will also automatically label PRs based on their code size. The following
# labels are given based on the total change numbers (addition + deletions on github)
# at the time the PR is opened (not updated as changes are made):
# Extra Small (XS): < 10 total changes.
# Small (S):        < 50 total changes.
# Medium (M):       < 250 total changes.
# Large (L):        < 1000 total changes.
# Extra Large (XL): >= 1000 total changes.
#
# Usage:
#     export CIRQ_BOT_GITHUB_ACCESS_TOKEN=[access token for CirqBot's github account]
#     export CIRQ_BOT_UPDATE_BRANCH_COOKIE=[CirqBot user_session cookie from github]
#     bash dev_tools/pr_monitor.sh
########################################################################################


# Get the working directory to the repo root.
thisdir="$(dirname "${BASH_SOURCE[0]}")" || exit $?
repo_dir="$(git -C "${thisdir}" rev-parse --show-toplevel)" || exit $?
cd "${repo_dir}" || exit $?

# Do the thing.
export PYTHONPATH=${repo_dir}
python3 "${repo_dir}/dev_tools/pr_monitor.py" "$@"
