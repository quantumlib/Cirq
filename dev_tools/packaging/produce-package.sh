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

set -o errexit
set -o nounset

trap "{ echo -e '\033[31mFAILED\033[0m'; }" ERR

DOC="\
usage: $0 [options] output_dir [version]

Produces wheels that can be uploaded to the pypi package repository.

First argument must be the output directory.  Second argument is an optional
version specifier, which overwrites the version in _version.py files.
If not set, the version from _version.py is used as is.

Options:

  -c, --commit=COMMIT   create wheels from sources at COMMIT instead of HEAD
  -h, --help            display this message and exit.
"

out_dir=""
specified_version=""
commitish=HEAD


die() {
    ( shift; echo -e "\033[31m${*}\033[0m" )
    exit "$1"
}

# Helper to isolate dev_tools/modules.py from Python environment variables
my_dev_tools_modules() {
    python3 -E dev_tools/modules.py "$@"
}

# Process command-line arguments
while (( $# )); do
    case "$1" in
        -h|--help)
            echo "$DOC"
            exit 0
            ;;
        -c?*)
            commitish="${1#-c}"
            ;;
        --commit=?*)
            commitish="${1#*=}"
            ;;
        -c|--commit)
            shift
            (( $# )) || die 2 "Option '-c,--commit' requires an argument."
            commit="$1"
            ;;
        *)
            if [[ -z "${out_dir}" ]]; then
                out_dir=$(realpath "${1}")
            elif [[ -z "${specified_version}" ]]; then
                specified_version="${1}"
            else
                die 2 "Unrecognized argument '$1'."
            fi
            ;;
    esac
    shift
done

if [[ -z "${out_dir}" ]]; then
    die 2 "No output directory given."
fi

# Change to the root of the Cirq git repository
thisdir=$(dirname "${BASH_SOURCE[0]:?}")
repo_dir=$(git -C "${thisdir}" rev-parse --show-toplevel)
cd "${repo_dir}"

# Validate and resolve the commit value
commit=$(git rev-parse --verify --quiet "${commitish}^{commit}") ||
    die "$?" "Invalid commit identifier '${commitish}'"

# Make a pristine temporary clone of the Cirq repository
if [[ -n "$(git status --short)" ]]; then
    echo -e "\033[31mWARNING: You have uncommitted changes. They won't be included in the package.\033[0m"
fi

tmp_git_dir=$(mktemp -d "/tmp/produce-package-git.XXXXXXXXXXXXXXXX")
echo "Creating pristine repository clone at ${tmp_git_dir}"
git clone --shared --quiet --no-checkout "${repo_dir}" "${tmp_git_dir}"
cd "${tmp_git_dir}"
git checkout --quiet "${commit}"

if [[ -n "${specified_version}" ]]; then
    current_version=$(my_dev_tools_modules print_version)
    my_dev_tools_modules replace_version --old="${current_version}" --new="${specified_version}"
fi

# Python 3 wheel.
echo "Producing Python 3 package files."

# Reuse SOURCE_DATE_EPOCH if specified in the caller environment
date_epoch=${SOURCE_DATE_EPOCH:-$(git log -1 --pretty="%ct")}
cirq_modules=$(my_dev_tools_modules list --mode folder --include-parent)

for m in ${cirq_modules}; do
    echo "creating wheel for ${m}"
    SOURCE_DATE_EPOCH="${date_epoch}" \
        python3 -m pip wheel --no-deps --wheel-dir="${out_dir}" "./${m}"
done

ls "${out_dir}"

# Final clean up (all is well if we got here)
cd "${repo_dir}"
rm -rf "${tmp_git_dir}"
