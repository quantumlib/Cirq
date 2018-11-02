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

# Uses the 3to2 tool to automatically translate cirq's python 3 code into
# python 2 code. Code is read from the given input directory (second command
# line argument) and written to the given output directory (first command line
# argument). The input directory defaults to the current working directory. The
# output directory defaults to "python2.7-output" in the current working
# directory.

# If compiled protobuffer files are desired, the protocol buffer compiler
# (protoc) must be installed. Instructions for installing the protocol buffer
# compiler can be found at
# https://github.com/protocolbuffers/protobuf/blob/master/src/README.md

# Takes three optional arguments:
#     1. Output directory. Defaults to 'python2.7-output'. Fails if the
#         directory already exists.
#     2. Input directory. Defaults to the current directory.
#     3. Path to the virtual environment to use (for 3to2). Defaults to the
#         current environment (or none). (This argument is used by scripts that
#         create test environments.)
#
# Example usage:
#    dev_tools/python2.7-generate.sh /tmp/converted-code ./ ~/virtual-envs/cirq3

set -e

out_dir=${1:-"$(pwd)/python2.7-output"}
in_dir=${2:-$(pwd)}
venv_dir=${3:-""}
project_name="cirq"

if [ -z "${venv_dir}" ]; then
  three_to_two_path="3to2"
else
  three_to_two_path="${venv_dir}/bin/3to2"
fi
if [ -z "${in_dir}" ]; then
  echo -e "\e[31mNo input directory given.\e[0m"
  exit 1
fi
if [ -z "${out_dir}" ]; then
  echo -e "\e[31mNo output directory given.\e[0m"
  exit 1
fi

mkdir ${out_dir}

trap "manual_cancel=1; exit 1" INT
function print_cached_err () {
  if [ -z "${manual_cancel}" ]; then
    cat "${out_dir}/err_tmp.log" 1>&2
  fi
  rm -rf "${out_dir}"
}
touch "${out_dir}/err_tmp.log"
trap print_cached_err ERR

# Copy into output directory and convert in-place.
cp -r "${in_dir}/${project_name}" "${out_dir}/${project_name}"
cp -r "${in_dir}/docs" "${out_dir}/docs"
"${three_to_two_path}" --nofix=numliterals --no-diffs --write --processes=16 "${out_dir}" >/dev/null 2> "${out_dir}/err_tmp.log"
find "${out_dir}" | grep "\.py\.bak$" | xargs rm -f

# Move examples without conversion (they must be cross-compatible).
cp -r "${in_dir}/examples" "${out_dir}/examples"

# Build protobufs if protoc is installed, otherwise show warning.
if command -v "protoc" > /dev/null; then
  proto_dir="${out_dir}/${project_name}/api/google/v1"
  find ${proto_dir} | grep '_pb2\.py' | xargs rm -f
  protoc -I="${out_dir}" --python_out="${out_dir}" ${proto_dir}/*.proto
else
  echo -e "\e[33mProtocol buffers were not compiled because protoc is not installed.\e[0m"
  echo -e "\e[33mSee https://github.com/protocolbuffers/protobuf/blob/master/src/README.md for protoc installation instructions.\e[0m"
fi

# Include requirements files.
cp "${in_dir}/dev_tools/python2.7-requirements.txt" "${out_dir}/requirements.txt"
cp "${in_dir}/dev_tools/conf/pip-list-python2.7-test-tools.txt" "${out_dir}/pip-list-test-tools.txt"

# Include packaging files.
cp "${in_dir}/MANIFEST.in" "${out_dir}/MANIFEST.in"
cp "${in_dir}/README.rst" "${out_dir}/README.rst"
cp "${in_dir}/LICENSE" "${out_dir}/LICENSE"
cp "${in_dir}/setup.py" "${out_dir}/setup.py"

# Mark every file as using utf8 encoding.
files_to_update=$(find ${out_dir} | grep "\.py$" | grep -v "_pb2\.py$")
for file in ${files_to_update}; do
    sed -i '1s/^/# coding=utf-8\n/' ${file}
done

# Whenever a __str__ method is defined, delegate to __unicode__.
for file in ${files_to_update}; do
    sed -i "s/^\(\s\+\?\)def __str__(self):/\1def __str__(self):\n\1    return unicode(self).encode('utf-8')\n\n\1def __unicode__(self):/" ${file}
done

# Replace itertools.zip_longest with itertools.izip_longest.
for file in ${files_to_update}; do
    sed -i "s/\bitertools.zip_longest\b/itertools.izip_longest/g" ${file}
done

rm -f "${out_dir}/err_tmp.log"
