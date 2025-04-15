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

[[ "${BASH_VERSINFO[0]}" -ge 4 ]] || { echo "ERROR: Bash version 4+ required." >&2; exit 1; }

set -euo pipefail -o errtrace
shopt -s inherit_errexit

declare -r usage="Usage:

${0##*/} [-h | --help | help] [--pr NUMBER] [--repo REPO] [--token TOKEN]

Updates the size labels on a pull request based on the number of lines it
changes. The pull request is identified by the value given after the option
'--pr' on the command line; if this option is not provided, the value of the
environment variable PR_NUMBER is used instead. The repository and access
tokens are determined by the values given to the options '--repo' and
'--token', respectively, or if those are not given on the command line, the
environment variables GITHUB_REPOSITORY and GITHUB_TOKEN, respectively.

Running this program with the option '-h', '--help', or 'help' will make it
print this help text and exit with exit code 0 without doing anything else."

declare -ar LABELS=(
    "Size: XS"
    "size: S"
    "size: M"
    "size: L"
    "size: XL"
)

declare -A LIMITS=(
    ["${LABELS[0]}"]=10
    ["${LABELS[1]}"]=50
    ["${LABELS[2]}"]=200
    ["${LABELS[3]}"]=800
    ["${LABELS[4]}"]="$((2 ** 63 - 1))"
)

declare -ar IGNORED=(
    "*_pb2.py"
    "*_pb2.pyi"
    "*_pb2_grpc.py"
    ".*.lock"
    "*.bundle.js"
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
    jq "${@}" 2> >(awk -v h="stderr from jq ${*}:" '{print h, $0}' 1>&2) &&
        rc="${?}" ||
        rc="${?}"
    if [[ "${rc}" != "0" ]]; then
        error "The jq program failed: ${*}"
        error "Note the quotes above may be wrong. Here was the (possibly empty) input in ${*: -1}:"
        cat "${@: -1}" # Assumes last argument is input file!!
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

function check_arg_value() {
    if [[ -z "$2" || "$2" == -* ]]; then
        error "Argument value for $1 is missing or looks like another option: '$2'"
        exit 1
    fi
}

function parse_options() {
    local pr="" repo="" token=""
    # Temporarily disable -u flag; this makes the code below much simpler.
    set +u
    while (( $# > 0 )); do
        case $1 in
            -h | --help | help)
                echo "$usage"
                exit 0
                ;;
            --pr)
                check_arg_value "$1" "$2"
                pr="$2"
                shift 2
                ;;
            --repo)
                check_arg_value "$1" "$2"
                repo="$2"
                shift 2
                ;;
            --token)
                check_arg_value "$1" "$2"
                token="$2"
                shift 2
                ;;
            -*)
                error "Unrecognized option $1."
                echo "$usage"
                exit 1
                ;;
            *)
                error "Too many arguments."
                echo "$usage"
                exit 1
                ;;
        esac
    done
    PR_NUMBER="${pr:-$PR_NUMBER}"
    GITHUB_REPOSITORY="${repo:-$GITHUB_REPOSITORY}"
    GITHUB_TOKEN="${token:-$GITHUB_TOKEN}"
    # Reinstate error-exit for undefined variables.
    set -u
}

function main() {
    local moreinfo="(Use --help option for more info.)"
    if [[ -z "$PR_NUMBER" ]]; then
        error "Missing pull request number. $moreinfo"
        exit 1
    fi
    if [[ -z "$GITHUB_REPOSITORY" ]]; then
        error "Missing repository identification. $moreinfo"
        exit 1
    fi
    if [[ -z "$GITHUB_TOKEN" ]]; then
        error "Missing GitHub access token. $moreinfo"
        exit 1
    fi

    local total_changes
    total_changes="$(compute_changes "$PR_NUMBER")"
    info "Lines changed: ${total_changes}"

    local size_label
    size_label="$(get_size_label "$total_changes")"
    info "Appropriate label is '${size_label}'"

    local correctly_labeled
    correctly_labeled="$(prune_stale_labels "$PR_NUMBER" "${size_label}")"

    if [[ "${correctly_labeled}" != true ]]; then
        api_call "issues/$PR_NUMBER/labels" -X POST -d "{\"labels\":[\"${size_label}\"]}" &>/dev/null
        info "Added label '${size_label}'"
    fi
}

parse_options "$@"
main
