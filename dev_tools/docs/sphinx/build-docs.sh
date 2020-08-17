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

################################################################################
# Generates a local copy of Cirq's documentation, using Sphinx.
#
# Output currently goes into [REPO_ROOT]/rtd-docs/_build
#
# Temporary files generated by Sphinx (e.g. by the Napoleon extension
# translating google-style docstrings) are put into [REPO_ROOT]/docs/generated,
# which is cleared before and after this command runs.
#
# Usage:
#     dev_tools/docs/sphinx/build-docs.sh
################################################################################

set -e
trap "{ echo -e '\033[31mFAILED\033[0m'; }" ERR

# Get the working directory to the repo root.
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$(git rev-parse --show-toplevel)"

docs_conf_dir="rtd-docs"
out_dir="rtd-docs/_build"
new_docs_dir="docs"

# Cleanup pre-existing temporary generated files.
rm -rf "${docs_conf_dir}/generated"

# Cleanup previous output.
rm -rf "${out_dir}"

# Stage files from the new structure to the ReadTheDocs one. 
# This should go away as QPolish gets published (ETA October 2020)

cp -R ${new_docs_dir} ${docs_conf_dir}/

# Regenerate docs.
sphinx-build -M html "${docs_conf_dir}" "${out_dir}" -W --keep-going -j auto

# Cleanup newly generated temporary files.
rm -rf "${docs_conf_dir}/generated"

rm -rf ${docs_conf_dir}/docs

echo
echo Index Page:
echo "file://$(pwd)/${out_dir}/html/index.html"
echo
