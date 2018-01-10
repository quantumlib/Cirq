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

# Uses the 3to2 tool to automatically translate cirq's python 3 code into
# python 2 code. Output goes into the directory 'python2.7-generated'. If that
# directory already exists, it will be deleted first.

# Assumes the current working directory is the root of cirq's git repository.
# Can be passed the directory of the output file, which will be overwritten,
# otherwise this defaults to python2.7-output.

set -e

out=${1:-'python2.7-output'}

rm -rf ${out}
mkdir ${out}
cp -r cirq ${out}/cirq
3to2 ${out}/cirq -w > /dev/null 2> /dev/null
find ${out}/cirq | grep "\.py\.bak$" | xargs rm -f

proto_dir=${out}/cirq/apis/google
find ${proto_dir} | grep '_pb2\.py' | xargs rm -f
protoc -I=${proto_dir} --python_out=${proto_dir} ${proto_dir}/*.proto

cp python2.7-requirements.txt ${out}/requirements.txt
