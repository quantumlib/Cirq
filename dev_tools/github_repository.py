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


class GithubRepository:
    """Details how to access a repository on github."""
    def __init__(self,
                 organization: str,
                 name: str,
                 access_token: Optional[str]) -> None:
        """
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
        return 'git@github.com:{}/{}.git'.format(self.organization,
                                                 self.name)
