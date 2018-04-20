#!/usr/bin/env bash

# Copyright 2018 Google LLC
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

set -e

input_dir=${1}
out=${2}

if [ -z "${input_dir}" ]; then
  echo -e "\e[31mNo input directory given.\e[0m"
  exit 1
fi
if [ -z "${out}" ]; then
  echo -e "\e[31mNo output directory given.\e[0m"
  exit 1
fi

mkdir ${out}

# Copy into output directory and convert in-place.
cp -r ${input_dir}/cirq ${out}/cirq
3to2 ${out}/cirq -w >/dev/null
find ${out}/cirq | grep "\.py\.bak$" | xargs rm -f

# Build protobufs.
proto_dir=${out}/cirq/api/google/v1
find ${proto_dir} | grep '_pb2\.py' | xargs rm -f
protoc -I=${out} --python_out=${out} ${proto_dir}/*.proto

cp ${input_dir}/python2.7-requirements.txt ${out}/requirements.txt
cp ${input_dir}/README.md ${out}/README.md

# Mark every file as using utf8 encoding.
files_to_update=$(find ${out} | grep "\.py$" | grep -v "_pb2\.py$")
for file in ${files_to_update}; do
    sed -i '1s/^/# coding=utf-8\n/' ${file}
done

# Whenever a __str__ method is defined, delegate to __unicode__.
for file in ${files_to_update}; do
      sed -i "s/^\(\s\+\?\)def __str__(self):/\1def __str__(self):\n\1    return unicode(self).encode('utf-8')\n\n\1def __unicode__(self):/" ${file}
done
