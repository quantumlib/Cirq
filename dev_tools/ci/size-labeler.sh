#!/usr/bin/env bash
# Copyright 2025 The Cirq Developers
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

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Updates size labels on pull requests based on the number of lines changed.
# Usage:
#   label-pr-size.sh THE_PR_NUMBER
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

[[ "${BASH_VERSINFO[0]}" -ge 4 ]] || { echo "ERROR: Bash version 4+ required." >&2; exit 1; }

set -euo pipefail -o errtrace
shopt -s inherit_errexit

declare -a LABELS=(
    "Size: XS"
    "Size: S"
    "Size: M"
    "Size: L"
    "Size: XL"
)

declare -A LIMITS=(
    ["${LABELS[0]}"]=10
    ["${LABELS[1]}"]=50
    ["${LABELS[2]}"]=200
    ["${LABELS[3]}"]=800
    ["${LABELS[4]}"]="$((2 ** 63 - 1))"
)

declare -a IGNORED=(
    "*_pb2.py"
    "*_pb2.pyi"
    "*_pb2_grpc.py"
    ".*.lock"
)

function info() {
    echo >&2 "INFO: ${*}"
}

function error() {
    echo >&2 "ERROR: ${*}"
}

function jq_stdin() {
    local infile
    infile="$(mktemp)"
    readonly infile

    cat >"${infile}"
    jq_file "$@" "${infile}"
}

function jq_file() {
    # Regardless of the success, store the return code.
    # Prepend each sttderr with command args and send back to stderr.
    jq "${@}" 2> >(echo "$(date -Iseconds) stderr from jq ${*}: " 1>&2) &&
        rc="${?}" ||
        rc="${?}"
    if [[ "${rc}" != "0" ]]; then
        error "The jq program failed: ${*}"
        error "Note the quotes above may be wrong. Here was the (possibly empty) input in ${*: -1}:"
        cat "${@: -1}" # Assumes last argument is input file!!
        exit 1
    fi
    return "${rc}"
}

function api_call() {
    local -r endpoint="${1// /%20}" # love that our label names have spaces...
    local -r uri="https://api.github.com/repos/${GITHUB_REPOSITORY}"
    info "Calling: ${uri}/${endpoint}"
    curl -sSL \
        -H "Authorization: token ${GITHUB_TOKEN}" \
        -H "Accept: application/vnd.github.v3.json" \
        -H "X-GitHub-Api-Version:2022-11-28" \
        -H "Content-Type: application/json" \
        "${@:2}" \
        "${uri}/${endpoint}"
}

function compute_changes() {
    local -r pr="$1"

    local change_info
    local -r keys_filter='with_entries(select([.key] | inside(["changes", "filename"])))'
    change_info="$(jq_stdin "map(${keys_filter})" <<<"$(api_call "pulls/${pr}/files")")"

    local files total_changes
    readarray -t files < <(jq_stdin -c '.[]' <<<"${change_info}")
    total_changes=0
    for file in "${files[@]}"; do
        local name changes
        name="$(jq_stdin -r '.filename' <<<"${file}")"
        for pattern in "${IGNORED[@]}"; do
            if [[ "$name" =~ ${pattern} ]]; then
                info "File $name ignored"
                continue 2
            fi
        done
        changes="$(jq_stdin -r '.changes' <<<"${file}")"
        info "File $name +-$changes"
        total_changes="$((total_changes + changes))"
    done
    echo "$total_changes"
}

function get_size_label() {
    local -r changes="$1"
    for label in "${LABELS[@]}"; do
        local limit="${LIMITS["${label}"]}"
        if [[ "${changes}" -lt "${limit}" ]]; then
            echo "${label}"
            return
        fi
    done
}

function prune_stale_labels() {
    local -r pr="$1"
    local -r size_label="$2"
    local existing_labels
    existing_labels="$(jq_stdin -r '.labels[] | .name' <<<"$(api_call "pulls/${pr}")")"
    readarray -t existing_labels <<<"${existing_labels}"

    local correctly_labeled=false
    for label in "${existing_labels[@]}"; do
        [[ -z "${label}" ]] && continue
        # If the label we want is already present, we can just leave it there.
        if [[ "${label}" == "${size_label}" ]]; then
            info "Label '${label}' is correct, leaving it."
            correctly_labeled=true
            continue
        fi
        # If there is another size label, we need to remove it
        if [[ -v "LIMITS[${label}]" ]]; then
            info "Label '${label}' is stale, removing it."
            api_call "issues/${pr}/labels/${label}" -X DELETE &>/dev/null
            continue
        fi
        info "Label '${label}' is unknown, leaving it."
    done
    echo "${correctly_labeled}"
}

function main() {
    local pr=$1
    info "Labeling PR ${pr}"

    local total_changes
    total_changes="$(compute_changes "${pr}")"
    info "Lines changed: ${total_changes}"

    local size_label
    size_label="$(get_size_label "$total_changes")"
    info "Appropriate label is '${size_label}'"

    local correctly_labeled
    correctly_labeled="$(prune_stale_labels "${pr}" "${size_label}")"

    if [[ "${correctly_labeled}" != true ]]; then
        api_call "issues/${pr}/labels" -X POST -d "{\"labels\":[\"${size_label}\"]}" &>/dev/null
        info "Added label '${size_label}'"
    fi
}

[[ "$#" -eq 1 ]] || { error "Missing required argument: the PR number." >&2; exit 1; }
[[ -v GITHUB_TOKEN ]] || { error "Variable GITHUB_TOKEN is not set."; exit 1; }
[[ -v GITHUB_REPOSITORY ]] || { error "Variable GITHUB_REPOSITORY is not set."; exit 1; }

main "$@"
