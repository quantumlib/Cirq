#!/bin/bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This simple script runs a local instance of Triage Party, for testing purposes.
# It follows examples given in https://github.com/google/triage-party#readme

function error() {
    echo -e >&2 "\033[31mError: ${*}.\033[0m"
}

if [[ -z "${GITHUB_TOKEN}" ]]; then
    error "GITHUB_TOKEN environment variable is not set"
    exit 1
fi
if ! command -v podman &> /dev/null; then
    error "cannot find podman"
    exit 1
fi

# Change the working directory to the repo root.
thisdir=$(dirname "${BASH_SOURCE[0]:?}") || exit $?
repo_dir=$(git -C "${thisdir}" rev-parse --show-toplevel) || exit $?
cd "${repo_dir}" || exit $?

kube_path="${PWD}/dev_tools/triage-party/kubernetes"

podman run \
     --rm \
     -p 8080:8080 \
     -e GITHUB_TOKEN \
     -v "${kube_path}/02_deployment/config.yaml:/app/config/config.yaml" \
     -v "${kube_path}/02_deployment/custom.css:/app/site/static/css/custom.css" \
     -v "${kube_path}/02_deployment/cirq-icon-very-small.png:/app/site/static/img/favicon-32x32.png" \
     docker.io/triageparty/triage-party
