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

from typing import Dict, Tuple, List, cast, Set, Optional

import os.path
import re

from dev_tools import env_tools, shell_tools

IGNORED_FILE_PATTERNS = [
    r'^dev_tools/.+',  # Environment-heavy code.
    r'^rtd_docs/.+',  # Environment-heavy code.
    r'^.+_pb2(_grpc)?\.py$',  # Auto-generated protobuf code.
    r'^(.+/)?setup\.py$',  # Installation code.
    r'^(.+/)?_version\.py$',  # Installation code.
    r'^cirq-google/cirq_google/cloud/.+\.py$',  # Generated gRPC client code.
    r'^cirq-google/cirq_google/api/v1/.+\.py$',  # deprecated API code
    r'^benchmarks/',
]
IGNORED_BLOCK_PATTERNS = [r'^\s*if TYPE_CHECKING:$']  # imports needed only while type-checking.
IGNORED_LINE_PATTERNS = [
    # Imports often uncovered due to version checks and type checking blocks.
    r'^import .+$',
    r'^from .+ import .+$',
    # Else lines are redundant w.r.t. other lines.
    r'^else:$',
    # Lines that are only invoked when running the file as the main file.
    r'^main\(.+$',
    r'^app.run\(main\)$',
    # Empty method definitions.
    r'^pass$',
    # Code explicitly marked as not implemented.
    r'raise NotImplementedError.*',
    # Code explicitly marked as unreachable.
    r'^assert False.*$',
    # Code testing if libraries are present.
    r'except ImportError',
    # Plotting code.
    r'plt\.show\(\)',
    r'fig(ure)?\.show\(\)',
    r'=\s*plt.subplots?\(',
    # Body of mypy Protocol methods.
    r'\.\.\.',
]
EXPLICIT_OPT_OUT_PATTERN = r'#\s*pragma:\s*no cover\s*$'


def diff_to_new_interesting_lines(unified_diff_lines: List[str]) -> Dict[int, str]:
    """Extracts a set of 'interesting' lines out of a GNU unified diff format.

    Format:
      gnu.org/software/diffutils/manual/html_node/Detailed-Unified.html

      @@ from-line-numbers to-line-numbers @@
        line-from-either-file
        ...
      @@ start,count start,count @@
        line-from-either-file
        ...
      @@ single start,count @@
        line-from-either-file
        ...
    Examples:
      Deleted line (5 is the deleted LOC, 7 is the guessed would-have-been loc
      in the updated file given other changes pushing the line around):
        @@ 5 7,0 @@
         - content-of-line

      Added line:
        @@ 5,0 7 @@
         + content-of-line

      Modified chunk:
        @@ 10,15 11,5 @@
         - removed-line
         + added-line
         ...

    Args:
        unified_diff_lines: Lines of output from git diff.
    Returns:
        A dictionary of "touched lines", with key equal to the line number and
        value equal to the reason the line was touched. Includes added lines
        and lines near changes (including removals).
    """
    interesting_lines = dict()
    for diff_line in unified_diff_lines:
        # Parse the 'new file' range parts of the unified diff.
        if not diff_line.startswith('@@ '):
            continue
        change = diff_line[3 : diff_line.index(' @@', 3)]
        new = change.split(' ')[1]
        start = int(new.split(',')[0])
        count = 1 if ',' not in new else int(new.split(',')[1])

        # The lines before and after a deletion should still be covered.
        if count == 0:
            for i in range(start, start + 2):
                interesting_lines[i] = 'is near a removal'
        else:
            for i in range(start, start + count):
                interesting_lines[i] = 'is new or changed'
    return interesting_lines


def fix_line_from_coverage_file(line):
    line = line.rstrip()
    if line.startswith('!'):
        line = line[1:]
    return line


def get_incremental_uncovered_lines(
    abs_path: str, base_commit: str, actual_commit: Optional[str]
) -> List[Tuple[int, str, str]]:
    """Find touched but uncovered lines in the given file.

    Uses git diff and the annotation files created by `pytest --cov-report annotate` to find
    these lines in the given file.

    Args:
        abs_path: The path of a file to look for uncovered lines in.
        base_commit: Old state to diff against.
        actual_commit: Current state. Use None to use local uncommitted files.

    Returns:
        A list of the indices, content, and reason-for-including of
        'interesting' uncovered lines. An interesting uncovered line is one
        involved with the diff.
    """
    # Deleted files don't have any lines that need to be covered.
    if not os.path.isfile(abs_path):
        return []

    optional_actual_commit = [] if actual_commit is None else [actual_commit]
    unified_diff_lines_str = shell_tools.output_of(
        ['git', 'diff', '--unified=0', base_commit, *optional_actual_commit, '--', abs_path]
    )
    unified_diff_lines = [e for e in unified_diff_lines_str.split('\n') if e.strip()]

    touched_lines = diff_to_new_interesting_lines(unified_diff_lines)

    with open(abs_path, 'r') as actual_file:
        ignored_lines = determine_ignored_lines(actual_file.read())

    cover_path = abs_path + ',cover'
    has_cover_file = os.path.isfile(cover_path)
    content_file = cover_path if has_cover_file else abs_path
    with open(content_file, 'r') as annotated_coverage_file:
        return [
            (i, fix_line_from_coverage_file(line), touched_lines[i])
            for i, line in enumerate(annotated_coverage_file, start=1)
            if i in touched_lines and i not in ignored_lines
            if line_counts_as_uncovered(line, has_cover_file)
        ]


def line_content_counts_as_uncovered_manual(content: str) -> bool:
    """Whether the given content counts as uncovered.

    Args:
        content: A line with indentation and tail comments/space removed.

    Returns:
        Whether the line could be included in the coverage report.
    """
    # Omit empty lines.
    if not content:
        return False

    # Omit declarations.
    for keyword in ['def', 'class']:
        if content.startswith(keyword) and content.endswith(':'):
            return False

    # TODO: multiline comments, multiline strings, etc, etc.
    # Github issue: https://github.com/quantumlib/Cirq/issues/2968
    return True


def determine_ignored_lines(content: str) -> Set[int]:
    lines = content.split('\n')
    result: List[int] = []

    explicit_opt_out_regexp = re.compile(EXPLICIT_OPT_OUT_PATTERN)
    i = 0
    while i < len(lines):
        line = lines[i]

        if any(re.match(pat, line) for pat in IGNORED_BLOCK_PATTERNS):
            end = naive_find_end_of_scope(lines, i + 1)
            result.extend(range(i, end))
            i = end
        elif explicit_opt_out_regexp.match(line.strip()):
            # Ignore the rest of a block.
            end = naive_find_end_of_scope(lines, i)
            result.extend(range(i, end))
            i = end
        elif explicit_opt_out_regexp.search(line):
            # Ignore a single line.
            result.append(i)
            i += 1
        else:
            # Don't ignore.
            i += 1

    # Line numbers start at 1, not 0.
    return {e + 1 for e in result}


def naive_find_end_of_scope(lines: List[str], i: int) -> int:
    # TODO: deal with line continuations, which may be less indented.
    # Github issue: https://github.com/quantumlib/Cirq/issues/2968
    line = lines[i]
    indent = line[: len(line) - len(line.lstrip())]
    while i < len(lines) and (not lines[i].strip() or lines[i].startswith(indent)):
        i += 1
    return i


def line_counts_as_uncovered(line: str, is_from_cover_annotation_file: bool) -> bool:
    """Whether the line should count as covered or not.

    Args:
        line: The line of code (including coverage annotation).
        is_from_cover_annotation_file: Whether this line has been annotated.

    Returns:
        True if the line is counted as uncovered.
    """

    # Ignore this line?
    if is_from_cover_annotation_file:
        # Already covered, or the tool deemed it not relevant for coverage.
        if not line.startswith('! '):
            return False

        content = line[2:]
    else:
        content = line

    # Ignore surrounding whitespace.
    content = content.strip()

    # Ignore end-of-line comments.
    # TODO: avoid # in strings, etc.
    # Github issue: https://github.com/quantumlib/Cirq/issues/2968
    if '#' in content:
        content = content[: content.index('#')].strip()

    # Ignored line pattern?
    if any(re.search(pat, content) for pat in IGNORED_LINE_PATTERNS):
        return False

    return is_from_cover_annotation_file or line_content_counts_as_uncovered_manual(content)


def is_applicable_python_file(rel_path: str) -> bool:
    """Determines if a file should be included in incremental coverage analysis.

    Args:
        rel_path: The repo-relative file path being considered.

    Returns:
        Whether to include the file.
    """
    return rel_path.endswith('.py') and not any(
        re.search(pat, rel_path) for pat in IGNORED_FILE_PATTERNS
    )


def check_for_uncovered_lines(env: env_tools.PreparedEnv) -> int:
    # Build context from environment.
    changed_files = env.get_changed_files()

    # Find/print lines that were changed but aren't covered.
    uncovered_count = 0
    for changed_file in changed_files:
        if not is_applicable_python_file(changed_file):
            continue

        base_path = cast(str, env.destination_directory)
        uncovered_lines = get_incremental_uncovered_lines(
            os.path.join(base_path, changed_file), env.compare_commit_id, env.actual_commit_id
        )

        if uncovered_lines:
            uncovered_count += len(uncovered_lines)
            print(
                shell_tools.highlight(
                    f'************* {changed_file} ({len(uncovered_lines)} uncovered)',
                    color_code=shell_tools.RED,
                )
            )
        for index, line, reason in uncovered_lines:
            # pylint: disable=consider-using-f-string
            print(
                'Line {} {} but not covered: {}'.format(
                    shell_tools.highlight(str(index).rjust(4), color_code=shell_tools.BOLD),
                    reason,
                    shell_tools.highlight(line, color_code=shell_tools.YELLOW),
                )
            )

    # Inform of aggregate result.
    print()
    if uncovered_count:
        print(
            shell_tools.highlight(
                f'Found {uncovered_count} uncovered touched lines.', color_code=shell_tools.RED
            )
        )
    else:
        print(shell_tools.highlight('All touched lines covered', color_code=shell_tools.GREEN))
    print()
    return uncovered_count
