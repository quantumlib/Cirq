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

"""Helper utilities for working with text files."""

def read_file_filtered(filename, begin_skip, end_skip):
    """Return lines from a file, skipping lines between markers.

    The strings `begin_skip` and `end_skip` indicate the start & end markers.
    They must be alone on separate lines of the input. Testing is done with
    str.startwith(...), which allows `begin_skip` and `end_skip` to be only
    the first parts of the markers. Matching is case-sensitive.

    Important: the start & end markers must not be nested.

    Args:
        filename: the name or path string of the file to read and process.
        begin_skip: beginning text of a line indicating start of skipping.
        end_skip: beginning text of a line indicating end of skipping.
    Returns:
        the contents of `filename` as a string, minus lines bracketed by
        `begin_stip` and `end_skip`, inclusive.
    Raises:
        ValueError: if the file contains unbalanced markers.
    """

    with open(filename, encoding='utf-8') as input_file:
        file_lines = input_file.readlines()

    skip = False
    content = ''
    for line_num, line in enumerate(file_lines, start=1):
        if line.startswith(begin_skip):
            if skip:
                raise ValueError(f"[Line {line_num}] Encountered"
                                 f" '{begin_skip}' while already skipping.")
            skip = True
        elif line.startswith(end_skip):
            if not skip:
                raise ValueError(f"[Line {line_num}] Encountered '{end_skip}'"
                                 f" without a matching '{begin_skip}'.")
            skip = False
        elif not skip:
            content += line
    if skip:
        raise ValueError(f"Missing final {end_skip}.")
    return content
