# Copyright 2024 Scaleway
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
from .scaleway_client import QaaSClient


class ScalewaySession:
    def __init__(self, client: QaaSClient, id: str, name: str):
        self.__id = id
        self.__client = client
        self.__name = name

    @property
    def status(self) -> str:
        """Returns the current status of the device session.

        Returns:
            str: the current status of the session. Can be either: starting, runnng, stopping, stopped
        """
        dict = self.__client.get_session(session_id=self.__id)

        return dict.get("status", "unknown_status")

    @property
    def id(self) -> str:
        """The unique identifier of the device session.

        Returns:
            str: The UUID of the current session.
        """
        return self.__id

    @property
    def name(self) -> str:
        """Name of the device session.

        Returns:
            str: the name of session.
        """
        return self.__name

    def stop(self) -> None:
        """Stops to the running device session.
        All attached jobs and their results will are kept up to 7 days before total deletion.
        """
        self.__client.terminate_session(session_id=self.__id)

    def delete(self) -> None:
        """Immediately stop and delete to the running device session.
        All attached jobs and their results will be purged from Scaleway service.
        """
        self.__client.delete_session(session_id=self.__id)
