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

from __future__ import annotations

import os
import sys

import requests

from dev_tools import github_repository, shell_tools


class PreparedEnv:
    """Details of a local environment that has been prepared for use."""

    def __init__(
        self,
        github_repo: github_repository.GithubRepository | None,
        actual_commit_id: str | None,
        compare_commit_id: str,
        destination_directory: str | None,
        virtual_env_path: str | None,
    ) -> None:
        """Initializes a description of a prepared (or desired) environment.

        Args:
            github_repo: The github repository that the local environment
                corresponds to. Use None if the actual_commit_id corresponds
                to a commit that isn't actually on github.
            actual_commit_id: Identifies the commit that has been checked out
                for testing purposes. Use None for 'local uncommitted changes'.
            compare_commit_id: Identifies a commit that the actual commit can
                be compared against, e.g. when diffing for incremental checks.
            destination_directory: The location where the environment has been
                prepared. If the directory isn't prepared yet, this should be
                None.
            virtual_env_path: The location of the python virtual environment
                that has been prepared for use when testing. If the virtual
                environment is not prepared yet, this should be None.
        """
        self.repository = github_repo
        self.actual_commit_id = actual_commit_id
        self.compare_commit_id = compare_commit_id
        if self.compare_commit_id == self.actual_commit_id:
            self.compare_commit_id += '~1'

        self.destination_directory = destination_directory
        self.virtual_env_path = virtual_env_path

    def bin(self, program: str) -> str:
        if self.virtual_env_path is None:
            return program
        return os.path.join(self.virtual_env_path, 'bin', program)

    def report_status_to_github(
        self, state: str, description: str, context: str, target_url: str | None = None
    ):
        """Sets a commit status indicator on github.

        If not running from a pull request (i.e. repository is None), then this
        just prints to stderr.

        Args:
            state: The state of the status indicator.
                Must be 'error', 'failure', 'pending', or 'success'.
            description: A summary of why the state is what it is,
                e.g. '5 lint errors' or 'tests passed!'.
            context: The name of the status indicator, e.g. 'pytest' or 'lint'.
            target_url: Optional location where additional details about the
                status can be found, e.g. an online test results page.

        Raises:
            ValueError: Not one of the allowed states.
            OSError: The HTTP post request failed, or the response didn't have
                a 201 code indicating success in the expected way. This is actually
                an IOError, which is equal to an OSError.
        """
        if state not in ['error', 'failure', 'pending', 'success']:
            raise ValueError(f'Unrecognized state: {state!r}')

        if self.repository is None or self.repository.access_token is None:
            return

        print(repr(('report_status', context, state, description, target_url)), file=sys.stderr)

        payload = {'state': state, 'description': description, 'context': context}
        if target_url is not None:
            payload['target_url'] = target_url

        url = (
            f"https://api.github.com/repos/{self.repository.organization}/"
            f"{self.repository.name}/statuses/{self.actual_commit_id}?"
            f"access_token={self.repository.access_token}"
        )

        response = requests.post(url, json=payload)

        if response.status_code != 201:
            raise IOError(
                f'Request failed. Code: {response.status_code}. Content: {response.content!r}.'
            )

    def get_changed_files(self) -> list[str]:
        """Get the files changed on one git branch vs another.

        Returns:
            list[str]: File paths of changed files, relative to the git repo
                root.
        """
        optional_actual_commit_id = [] if self.actual_commit_id is None else [self.actual_commit_id]
        out = shell_tools.output_of(
            [
                'git',
                'diff',
                '--name-only',
                self.compare_commit_id,
                *optional_actual_commit_id,
                '--',
            ],
            cwd=self.destination_directory,
        )
        return [e for e in out.split('\n') if e.strip()]
