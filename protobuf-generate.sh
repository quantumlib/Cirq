#!/usr/bin/env bash

# Copyright 2017 Google LLC
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

# Uses the protobuf compiler to regenerate the python code in cirq/format from
# the .proto files in that same directory.

# Assumes the current working directory is the root of cirq's git repository.

set -e

dir='cirq/apis/google'

find ${dir} | grep '_pb2\.py' | xargs rm -f
protoc -I=${dir} --python_out=${dir} ${dir}/*.proto
