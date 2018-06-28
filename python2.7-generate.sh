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

set -e

out_dir=${1:-"$(pwd)/python2.7-output"}
in_dir=${2:-$(pwd)}

if [ -z "${in_dir}" ]; then
  echo -e "\e[31mNo input directory given.\e[0m"
  exit 1
fi
if [ -z "${out_dir}" ]; then
  echo -e "\e[31mNo output directory given.\e[0m"
  exit 1
fi

mkdir ${out_dir}

function print_cached_err () {
  cat "${out_dir}/err_tmp.log" 1>&2
  rm -rf "${out_dir}"
}
touch "${out_dir}/err_tmp.log"
trap print_cached_err ERR

# Copy into output directory and convert in-place.
cp -r "${in_dir}/cirq" "${out_dir}/cirq"
cp -r "${in_dir}/docs" "${out_dir}/docs"
cp -r "${in_dir}/examples" "${out_dir}/examples"
3to2 "${out_dir}" -w >/dev/null 2> "${out_dir}/err_tmp.log"
find "${out_dir}" | grep "\.py\.bak$" | xargs rm -f

# Build protobufs.
proto_dir="${out_dir}/cirq/api/google/v1"
find ${proto_dir} | grep '_pb2\.py' | xargs rm -f
protoc -I="${out_dir}" --python_out="${out_dir}" ${proto_dir}/*.proto

cp "${in_dir}/python2.7-runtime-requirements.txt" "${out_dir}/python2.7-runtime-requirements.txt"
cp "${in_dir}/python2.7-dev-requirements.txt" "${out_dir}/python2.7-dev-requirements.txt"
cp "${in_dir}/README.rst" "${out_dir}/README.rst"

# Mark every file as using utf8 encoding.
files_to_update=$(find ${out_dir} | grep "\.py$" | grep -v "_pb2\.py$")
for file in ${files_to_update}; do
    sed -i '1s/^/# coding=utf-8\n/' ${file}
done

# Whenever a __str__ method is defined, delegate to __unicode__.
for file in ${files_to_update}; do
      sed -i "s/^\(\s\+\?\)def __str__(self):/\1def __str__(self):\n\1    return unicode(self).encode('utf-8')\n\n\1def __unicode__(self):/" ${file}
done

rm -f "${out_dir}/err_tmp.log"
