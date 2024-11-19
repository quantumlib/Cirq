#!/usr/bin/env bash

# Copyright 2024 The Cirq Developers
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

# ─────────────────────────────────────────────────────────────────────────────
# Simple script to convert .rst files to .md files. Used to convert all .rst
# files that were originally used in the Cirq repo, and saved here both to
# document how it was done and in case it needs to be done again in the future.
# ─────────────────────────────────────────────────────────────────────────────

declare -r program=${0##*/}
declare -r output_format="gfm+tex_math_gfm"

set -o nounset -o pipefail
shopt -s nullglob

# ~~~~ Helpers ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Check if the given string is a common help argument. Usage: "is_help $arg".
function is_help() {
    local -r value="$1"
    local -a help_strings=("-h" "--help" "help")
    local result=1

    saved_IFS="$IFS"
    IFS=","
    # shellcheck disable=SC2199
    if [[ "${IFS}${help_strings[@]}${IFS}" =~ ${IFS}${value}${IFS} ]]; then
        # Note Bash return vals are reversed from Boolean true/false.
        result=0
    fi
    IFS="$saved_IFS"
    return $result
}

# Test if a command is available. Usage: "if have foo; then ...; fi"
function have() {
    unset -v have
    type "$1" &> /dev/null
}

# ~~~~ Main body ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if [ $# -eq 0 ] || is_help "$1"; then
    cat <<EOF >&2
Usage: $program [-h] [file ...]

This program takes one or more files in reStructuredText (.rst) format, and
converts them to GitHub-flavored Markdown (.md) format. The output files will
be named the same as the input files, with the suffix .rst changed to .md.
Pandoc is used to perform the conversion.
EOF
    exit 0
fi

if ! have pandoc; then
    echo "Unable to find program 'pandoc' – please check that it's installed."
    exit 1
fi

for file in "$@"; do
    if ! [ -e "$file" ]; then
        echo "Skipping file $file because it does not exist."
        continue
    fi
    if ! [ -r "$file" ]; then
        echo "Skipping file $file because it is not readable."
        continue
    fi

    extension="${file##*.}"
    if [[ "$extension" != "rst" ]]; then
        echo "Skipping file $file because the name does not end in .rst"
        continue
    fi

    outfile="${file%.*}.md"
    echo "Converting $file and writing output to $outfile"
    pandoc -f rst -t $output_format -i "$file" -o "$outfile"
done

echo "Done. Note: you may need to tweak the results manually."
