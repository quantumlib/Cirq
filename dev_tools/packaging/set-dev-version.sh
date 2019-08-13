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

set -e
trap "{ echo -e '\033[31mFAILED\033[0m'; set +e }" ERR

PROJECT_NAME=cirq

ACTUAL_VERSION_LINE=$(cat "${PROJECT_NAME}/_version.py" | tail -n 1)
ACTUAL_VERSION=`echo $ACTUAL_VERSION_LINE | cut -d'"' -f 2`

unset CIRQ_DEV_VERSION
if [[ ${ACTUAL_VERSION_LINE} == *"dev"* ]]; then
  export CIRQ_DEV_VERSION="${ACTUAL_VERSION}$(date "+%Y%m%d%H%M%S")"
fi

unset ACTUAL_VERSION_LINE
unset ACTUAL_VERSION

set +e