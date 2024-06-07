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
        dict = self.__client.get_session(session_id=self.__id)

        return dict.get("status", "unknown_status")

    @property
    def id(self) -> str:
        return self.__id

    @property
    def name(self) -> str:
        return self.__name

    def stop(self) -> None:
        self.__client.terminate_session(session_id=self.__id)

    def delete(self) -> None:
        self.__client.delete_session(session_id=self.__id)
