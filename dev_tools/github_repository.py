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

from typing import Optional

import requests


class GithubRepository:
    """Details how to access a repository on github."""

    def __init__(self, organization: str, name: str, access_token: Optional[str]) -> None:
        """Inits GithubRepository.

        Args:
            organization: The github organization the repository is under.
            name: The name of the github repository.
            access_token: If present, this token is used to authorize changes
                to the repository when calling the github API (e.g. set build
                status indicators). Avoid using access tokens with more
                permissions than necessary.
        """
        self.organization = organization
        self.name = name
        self.access_token = access_token

    def as_remote(self) -> str:
        """Returns a string identifying the location of this repository."""
        return f'git@github.com:{self.organization}/{self.name}.git'

    def delete(self, url, **kwargs):
        return requests.delete(url, **self._auth(kwargs))

    def get(self, url, **kwargs):
        return requests.get(url, **self._auth(kwargs))

    def put(self, url, **kwargs):
        return requests.put(url, **self._auth(kwargs))

    def post(self, url, **kwargs):
        return requests.post(url, **self._auth(kwargs))

    def patch(self, url, **kwargs):
        return requests.patch(url, **self._auth(kwargs))

    def _auth(self, kwargs):
        new_kwargs = kwargs.copy()
        headers = kwargs.get('headers', {})
        headers.update({"Authorization": f"token {self.access_token}"})
        new_kwargs.update(headers=headers)
        return new_kwargs
