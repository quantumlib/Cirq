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

from __future__ import annotations

import pathlib
from unittest import mock

import pytest

from dev_tools import path_area_labeler

LABELER_CONFIG = pathlib.Path(".github/labeler.yml")


def test_load_labeler_config_parses_rules() -> None:
    config = path_area_labeler.load_labeler_config(LABELER_CONFIG)
    assert config["area/gates"] == ["cirq-core/cirq/ops/**"]
    assert "cirq-google/cirq_google/engine/**" in config["area/google/engine"]


@pytest.mark.parametrize(
    "glob_pattern,path,expected",
    [
        ("cirq-core/cirq/ops/**", "cirq-core/cirq/ops/common_gates.py", True),
        ("cirq-core/cirq/ops/**", "cirq-core/cirq/sim/sparse_simulator.py", False),
        ("**/setup.py", "cirq-core/setup.py", True),
        ("docs/**", "docs/dev/triage.md", True),
    ],
)
def test_glob_matches(glob_pattern: str, path: str, expected: bool) -> None:
    assert path_area_labeler.glob_matches(glob_pattern, path) is expected


def test_labels_for_paths_applies_multiple_matches() -> None:
    config = path_area_labeler.load_labeler_config(LABELER_CONFIG)
    labels = path_area_labeler.labels_for_paths(
        config, ["cirq-core/cirq/ops/common_gates.py", "cirq-google/cirq_google/engine/engine.py"]
    )
    assert labels == {"area/gates", "area/google", "area/google/engine", "interface/cirq-google"}


def test_extract_paths_from_text_finds_backtick_and_plain_paths() -> None:
    text = """
    Bug in `cirq-core/cirq/sim/sparse_simulator.py` when calling gates.
    Also see docs/dev/triage.md for context.
    """
    paths = path_area_labeler.extract_paths_from_text(text)
    assert paths == {"cirq-core/cirq/sim/sparse_simulator.py", "docs/dev/triage.md"}


def test_labels_for_issue_text() -> None:
    config = path_area_labeler.load_labeler_config(LABELER_CONFIG)
    text = "Failure in cirq-ionq/cirq_ionq/service.py when submitting jobs."
    paths = path_area_labeler.extract_paths_from_text(text)
    labels = path_area_labeler.labels_for_paths(config, paths)
    assert labels == {"interface/cirq-ionq"}


def test_normalize_path() -> None:
    assert path_area_labeler.normalize_path(".\\docs\\foo.md") == "docs/foo.md"
    assert path_area_labeler.normalize_path("./docs/foo.md") == "docs/foo.md"


def test_load_labeler_config_ignores_comments_and_empty_rules(tmp_path: pathlib.Path) -> None:
    config_path = tmp_path / "labeler.yml"
    config_path.write_text(
        """
# comment

'area/test':
- changed-files:
  - any-glob-to-any-file: 'docs/**'

'area/empty':

'area/also-empty':
- changed-files:
""",
        encoding="utf-8",
    )
    config = path_area_labeler.load_labeler_config(config_path)
    assert config == {"area/test": ["docs/**"]}


def test_labels_for_paths_returns_empty_set_for_no_matches() -> None:
    config = path_area_labeler.load_labeler_config(LABELER_CONFIG)
    assert path_area_labeler.labels_for_paths(config, ["unknown/path.py"]) == set()


def test_glob_matches_directory_prefix_exact() -> None:
    assert path_area_labeler.glob_matches("docs/**", "docs")


def test_fetch_issue_text() -> None:
    with mock.patch("dev_tools.path_area_labeler.subprocess.run") as run:
        run.return_value = mock.Mock(stdout='{"title": "Bug", "body": "Details"}')
        text = path_area_labeler.fetch_issue_text(issue_number=1, repo="quantumlib/Cirq")
    assert text == "Bug\nDetails"
    run.assert_called_once()


def test_apply_issue_labels_skips_when_empty() -> None:
    with mock.patch("dev_tools.path_area_labeler.subprocess.run") as run:
        path_area_labeler.apply_issue_labels(issue_number=1, repo="quantumlib/Cirq", labels=[])
    run.assert_not_called()


def test_apply_issue_labels_calls_gh() -> None:
    with mock.patch("dev_tools.path_area_labeler.subprocess.run") as run:
        path_area_labeler.apply_issue_labels(
            issue_number=42, repo="quantumlib/Cirq", labels=["area/gates", "area/docs"]
        )
    run.assert_called_once_with(
        [
            "gh",
            "issue",
            "edit",
            "42",
            "--repo",
            "quantumlib/Cirq",
            "--add-label",
            "area/docs",
            "area/gates",
        ],
        check=True,
    )


def test_main_dry_run_prints_labels(
    capsys: pytest.CaptureFixture[str], tmp_path: pathlib.Path
) -> None:
    config_path = tmp_path / "labeler.yml"
    config_path.write_text(
        "'area/gates':\n- changed-files:\n  - any-glob-to-any-file: 'cirq-core/cirq/ops/**'\n",
        encoding="utf-8",
    )
    rc = path_area_labeler.main(
        ["--config", str(config_path), "--paths", "cirq-core/cirq/ops/x.py", "--dry-run"]
    )
    assert rc == 0
    assert capsys.readouterr().out.strip() == "area/gates"


def test_main_prints_labels_without_issue_number(capsys: pytest.CaptureFixture[str]) -> None:
    rc = path_area_labeler.main(["--text", "Bug in cirq-core/cirq/ops/x.py"])
    assert rc == 0
    assert "area/gates" in capsys.readouterr().out


def test_main_applies_issue_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_REPOSITORY", "quantumlib/Cirq")
    with (
        mock.patch.object(
            path_area_labeler, "fetch_issue_text", return_value="cirq-core/cirq/ops/x.py"
        ),
        mock.patch.object(path_area_labeler, "apply_issue_labels") as apply,
    ):
        rc = path_area_labeler.main(["--issue-number", "99"])
    apply.assert_called_once()
    assert rc == 0


def test_main_issue_number_without_matching_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_REPOSITORY", "quantumlib/Cirq")
    with (
        mock.patch.object(
            path_area_labeler, "fetch_issue_text", return_value="no repository paths"
        ),
        mock.patch.object(path_area_labeler, "apply_issue_labels") as apply,
    ):
        rc = path_area_labeler.main(["--issue-number", "99"])
    apply.assert_not_called()
    assert rc == 0


def test_main_requires_github_repository_for_issue_number() -> None:
    with pytest.raises(SystemExit, match="GITHUB_REPOSITORY"):
        path_area_labeler.main(["--issue-number", "1"])
