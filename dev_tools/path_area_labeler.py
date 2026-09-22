# Copyright 2026 The Cirq Developers
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

"""Match repository paths to labels defined in ``.github/labeler.yml``."""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

LABEL_LINE = re.compile(r"""^['"]([^'"]+)['"]:\s*$""")
GLOB_LINE = re.compile(r"""any-glob-to-any-file:\s*['"]([^'"]+)['"]""")
PATH_TOKEN = re.compile(
    r"(?<![\w./@-])"
    r"((?:\.github|cirq-(?:core|google|ionq|aqt|pasqal|web)|docs|dev_tools|examples|benchmarks|check)"
    r"(?:/[\w@.+~-]+)+"
    r"(?:\.\w+)?)"
)


def load_labeler_config(config_path: Path) -> dict[str, list[str]]:
    """Parse ``labeler.yml`` path rules into a label-to-globs mapping."""
    label_to_globs: dict[str, list[str]] = {}
    current_label: str | None = None

    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        label_match = LABEL_LINE.match(line)
        if label_match:
            current_label = label_match.group(1)
            label_to_globs.setdefault(current_label, [])
            continue

        glob_match = GLOB_LINE.search(line)
        if glob_match and current_label is not None:
            label_to_globs[current_label].append(glob_match.group(1))

    return {label: globs for label, globs in label_to_globs.items() if globs}


def normalize_path(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")


def glob_matches(glob_pattern: str, path: str) -> bool:
    """Return whether ``path`` matches a labeler glob pattern."""
    normalized_path = normalize_path(path)
    normalized_glob = glob_pattern.replace("\\", "/")

    if normalized_glob.endswith("/**"):
        prefix = normalized_glob[:-3]
        return normalized_path == prefix or normalized_path.startswith(f"{prefix}/")

    if "**" in normalized_glob:
        return fnmatch.fnmatchcase(normalized_path, normalized_glob)

    return fnmatch.fnmatchcase(normalized_path, normalized_glob)


def labels_for_paths(config: dict[str, list[str]], paths: Iterable[str]) -> set[str]:
    """Return labels whose glob rules match any of ``paths``."""
    normalized_paths = {normalize_path(path) for path in paths}
    matched_labels: set[str] = set()

    for label, globs in config.items():
        for path in normalized_paths:
            if any(glob_matches(glob_pattern, path) for glob_pattern in globs):
                matched_labels.add(label)
                break

    return matched_labels


def extract_paths_from_text(text: str) -> set[str]:
    """Extract repository path-like tokens from free-form issue text."""
    return {normalize_path(match.group(1)) for match in PATH_TOKEN.finditer(text)}


def fetch_issue_text(*, issue_number: int, repo: str) -> str:
    completed = subprocess.run(
        [
            "gh",
            "issue",
            "view",
            str(issue_number),
            "--repo",
            repo,
            "--json",
            "title,body",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    return f"{payload.get('title', '')}\n{payload.get('body', '')}"


def apply_issue_labels(*, issue_number: int, repo: str, labels: Iterable[str]) -> None:
    label_args = sorted(set(labels))
    if not label_args:
        return

    subprocess.run(
        [
            "gh",
            "issue",
            "edit",
            str(issue_number),
            "--repo",
            repo,
            "--add-label",
            *label_args,
        ],
        check=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply area/interface labels based on .github/labeler.yml rules."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(".github/labeler.yml"),
        help="Path to the labeler configuration file.",
    )
    parser.add_argument(
        "--text",
        help="Free-form text to scan for repository paths (for testing or dry runs).",
    )
    parser.add_argument(
        "--paths",
        nargs="*",
        default=(),
        help="Explicit repository paths to match against the labeler rules.",
    )
    parser.add_argument(
        "--issue-number",
        type=int,
        help="Issue number to fetch via gh and label in the current repository.",
    )
    parser.add_argument(
        "--repo",
        default=None,
        help="GitHub repository slug (owner/name). Defaults to GITHUB_REPOSITORY.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print matched labels without applying them to an issue.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_labeler_config(args.config)

    paths: set[str] = set(args.paths)
    if args.text:
        paths.update(extract_paths_from_text(args.text))

    if args.issue_number is not None:
        repo = args.repo or _require_env("GITHUB_REPOSITORY")
        issue_text = fetch_issue_text(issue_number=args.issue_number, repo=repo)
        paths.update(extract_paths_from_text(issue_text))

    labels = labels_for_paths(config, paths)
    if args.dry_run or args.issue_number is None:
        for label in sorted(labels):
            print(label)
        return 0

    if labels:
        repo = args.repo or _require_env("GITHUB_REPOSITORY")
        apply_issue_labels(issue_number=args.issue_number, repo=repo, labels=labels)
    return 0


def _require_env(name: str) -> str:
    import os

    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"Missing required environment variable: {name}")
    return value


if __name__ == "__main__":
    sys.exit(main())
