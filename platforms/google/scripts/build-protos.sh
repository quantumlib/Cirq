#!/usr/bin/env bash

# Copyright 2019 The Cirq Developers
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
# Generates python from protobuf definitions, including mypy stubs.
#
# Usage:
#     dev_tools/build-protos.sh
################################################################################

# TODO(balintp): move this to platforms/google

set -e
trap "{ echo -e '\033[31mFAILED\033[0m'; }" ERR

# Get the working directory to the google root.
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ".."

# Build protos for each protobuf package.
for package in cirq_google/api/v1 cirq_google/api/v2 cirq_google/engine/client/quantum_v1alpha1/proto
do
  python -m grpc_tools.protoc -I=. --python_out=. --mypy_out=. ${package}/*.proto
done
