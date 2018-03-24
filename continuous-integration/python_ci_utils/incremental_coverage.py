#!/usr/bin/env python

import subprocess
from typing import Dict, Tuple, List

import os.path
import re
import sys

from python_ci_utils import git_shell_tools

IGNORED_FILE_PATTERNS = [
    r'^.+_pb2(_grpc)?\.py$',  # Auto-generated protobuf code.
]
IGNORED_LINE_PATTERNS = [
    r'^else:$',  # Always redundant w.r.t. other lines.
]


def diff_to_new_interesting_lines(unified_diff_lines: List[str]
                                  ) -> Dict[int, str]:
    """
    Extracts a set of 'interesting' lines out of a GNU unified diff format.

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
        change = diff_line[3:diff_line.index(' @@', 3)]
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


def get_incremental_uncovered_lines(abs_path: str,
                                    base_commit: str,
                                    actual_commit: str
                                    ) -> List[Tuple[int, str, str]]:
    """
    Uses git diff and the annotation files created by
    `pytest --cov-report annotate` to find touched but uncovered lines in the
    given file.

    Args:
        abs_path: The path of a file to look for uncovered lines in.
        base_commit: Old state to diff against.
        actual_commit: Current state.

    Returns:
        A list of the indices, content, and reason-for-including of
        'interesting' uncovered lines. An interesting uncovered line is one
        involved with the diff.
    """
    # Deleted files don't have any lines that need to be covered.
    if not os.path.isfile(abs_path):
        return []

    unified_diff_lines = git_shell_tools.run_lines(
        'git',
        'diff',
        '--unified=0',
        base_commit,
        actual_commit,
        abs_path)

    touched_lines = diff_to_new_interesting_lines(unified_diff_lines)

    cover_path = abs_path + ',cover'
    has_cover_file = os.path.isfile(cover_path)
    content_file = cover_path if has_cover_file else abs_path
    with open(content_file, 'r') as annotated_coverage_file:
        return [(i, fix_line_from_coverage_file(line), touched_lines[i])
                for i, line in enumerate(annotated_coverage_file, start=1)
                if i in touched_lines
                if line_counts_as_uncovered(line, has_cover_file)]


def line_content_counts_as_uncovered_manual(content: str) -> bool:
    """
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
    return True


def line_counts_as_uncovered(line: str,
                             is_from_cover_annotation_file: bool) -> bool:
    """
    Args:
        line: The line of code (including coverage annotation).
        is_from_cover_annotation_file: Whether this line has been annotated.
    Returns:
        Does the line count as uncovered?
    """

    # Ignore this line?
    if is_from_cover_annotation_file:
        # Already covered, or the tool deemed it not relevant for coverage.
        if not line.startswith('! '):
            return False

        content = line[2:]
    else:
        content = line

    # Remove end-of-line comments and surrounding whitespace.
    content = content.strip()
    # TODO: avoid # in strings, etc.
    if '#' in content:
        content = content[:content.index('#')].strip()

    # Ignored line pattern?
    if any(re.search(pat, content) for pat in IGNORED_LINE_PATTERNS):
        return False

    return (is_from_cover_annotation_file or
            line_content_counts_as_uncovered_manual(content))


def is_applicable_python_file(rel_path: str) -> bool:
    """
    Determines if a file should be included in incremental coverage analysis.

    Args:
        rel_path: The repo-relative file path being considered.
    Returns:
        Whether to include the file.
    """
    return (rel_path.endswith('.py') and
            not any(re.search(pat, rel_path) for pat in IGNORED_FILE_PATTERNS))


def do_test(root: str, commit_id: str, compare_id: str) -> Tuple[int, str]:
    # Trigger pytest to produce coverage output.
    try:
        print('Running pytest with --cov...')
        git_shell_tools.run_forward_into_std(
            'pytest',
            os.path.join(root, 'cirq'),
            '--cov',
            '--cov-report=annotate')
    except subprocess.CalledProcessError:
        # The tests failed. But we can still try to compute coverage.
        pass
    print()

    # Build context from environment.
    changed_files = git_shell_tools.get_changed_files_between(
        compare_id, commit_id)

    # Find/print lines that were changed but aren't covered.
    uncovered_count = 0
    for changed_file in changed_files:
        if not is_applicable_python_file(changed_file):
            continue

        uncovered_lines = get_incremental_uncovered_lines(
            os.path.join(root, changed_file),
            compare_id,
            commit_id)

        if uncovered_lines:
            uncovered_count += len(uncovered_lines)
            print(git_shell_tools.highlight(
                '************* {} ({} uncovered)'.format(
                    changed_file,
                    len(uncovered_lines)),
                color_code=git_shell_tools.RED))
        for index, line, reason in uncovered_lines:
            print('Line {} {} but not covered: {}'.format(
                git_shell_tools.highlight(str(index).rjust(4),
                                          color_code=git_shell_tools.BOLD),
                reason,
                git_shell_tools.highlight(line,
                                          color_code=git_shell_tools.YELLOW)))

    # Inform of aggregate result.
    print()
    if uncovered_count:
        print(git_shell_tools.highlight(
            'Found {} uncovered touched lines.'.format(uncovered_count),
            color_code=git_shell_tools.RED))
    else:
        print(git_shell_tools.highlight('All touched lines covered',
                                        color_code=git_shell_tools.GREEN))
    print()
    return uncovered_count, commit_id


def main():
    state = 'pending'
    description = ''

    access_token = ''
    try:
        uncovered_count, commit_id = git_shell_tools.do_in_temporary_test_environment(
            do_test,
            repository_organization='quantumlib',
            repository_name='cirq')

        # Report result to github.
        if uncovered_count:
            state = 'failure'
            description = '{} changed lines not covered'.format(
                uncovered_count)
        else:
            state = 'success'
            description = 'All covered!'

    except:
        commit_id = None  # TODO: get this out somehow
        state = 'error'
        description = 'testing script failed'
        raise

    # git_shell_tools.github_set_status_indicator(
    #     repository_organization='quantumlib',
    #     repository_name='cirq',
    #     repository_access_token=access_token,
    #     commit_id=commit_id,
    #     state=state,
    #     description=description,
    #     context='pytest (manual)',
    #     target_url=None)


if __name__ == '__main__':
    main()
