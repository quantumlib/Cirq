# -*- coding: utf-8 -*-
#
# Copyright 2020 The Cirq Developers
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

"""Wrappers for protocol buffer enum types."""

import enum


class ExecutionStatus(object):

    class State(enum.IntEnum):
        """
        -

        Attributes:
          STATE_UNSPECIFIED (int): -
          READY (int): -
          RUNNING (int): -
          CANCELLING (int): -
          CANCELLED (int): -
          SUCCESS (int): -
          FAILURE (int): -
        """
        STATE_UNSPECIFIED = 0
        READY = 1
        RUNNING = 2
        CANCELLING = 3
        CANCELLED = 4
        SUCCESS = 5
        FAILURE = 6

    class Failure(object):

        class Code(enum.IntEnum):
            """
            -

            Attributes:
              CODE_UNSPECIFIED (int): -
              SYSTEM_ERROR (int): -
              INVALID_PROGRAM (int): -
              INVALID_RUN_CONTEXT (int): -
              READ_PROGRAM_NOT_FOUND_IN_GCS (int): -
              READ_PROGRAM_PERMISSION_DENIED (int): -
              READ_PROGRAM_UNKNOWN_ERROR (int): -
              READ_RUN_CONTEXT_NOT_FOUND_IN_GCS (int): -
              READ_RUN_CONTEXT_PERMISSION_DENIED (int): -
              READ_RUN_CONTEXT_UNKNOWN_ERROR (int): -
              WRITE_RESULT_ALREADY_EXISTS_IN_GCS (int): -
              WRITE_RESULT_GCS_PERMISSION_DENIED (int): -
              SCHEDULING_EXPIRED (int): -
            """
            CODE_UNSPECIFIED = 0
            SYSTEM_ERROR = 1
            INVALID_PROGRAM = 2
            INVALID_RUN_CONTEXT = 3
            READ_PROGRAM_NOT_FOUND_IN_GCS = 4
            READ_PROGRAM_PERMISSION_DENIED = 5
            READ_PROGRAM_UNKNOWN_ERROR = 6
            READ_RUN_CONTEXT_NOT_FOUND_IN_GCS = 7
            READ_RUN_CONTEXT_PERMISSION_DENIED = 8
            READ_RUN_CONTEXT_UNKNOWN_ERROR = 9
            WRITE_RESULT_ALREADY_EXISTS_IN_GCS = 10
            WRITE_RESULT_GCS_PERMISSION_DENIED = 11
            SCHEDULING_EXPIRED = 14


class QuantumProcessor(object):

    class Health(enum.IntEnum):
        """
        -

        Attributes:
          HEALTH_UNSPECIFIED (int): -
          OK (int): -
          DOWN (int): -
          UNAVAILABLE (int): -
        """
        HEALTH_UNSPECIFIED = 0
        OK = 1
        DOWN = 2
        UNAVAILABLE = 3


class QuantumTimeSlot(object):

    class TimeSlotType(enum.IntEnum):
        """
        -

        Attributes:
          TIME_SLOT_TYPE_UNSPECIFIED (int): -
          MAINTENANCE (int): -
          OPEN_SWIM (int): -
          RESERVATION (int): -
          UNALLOCATED (int): -
        """
        TIME_SLOT_TYPE_UNSPECIFIED = 0
        MAINTENANCE = 1
        OPEN_SWIM = 2
        RESERVATION = 3
        UNALLOCATED = 4


class StreamError(object):

    class Code(enum.IntEnum):
        """
        -

        Attributes:
          CODE_UNSPECIFIED (int): -
          INTERNAL (int): -
          INVALID_ARGUMENT (int): -
          PERMISSION_DENIED (int): -
          PROGRAM_ALREADY_EXISTS (int): -
          JOB_ALREADY_EXISTS (int): -
          PROGRAM_DOES_NOT_EXIST (int): -
          JOB_DOES_NOT_EXIST (int): -
          PROCESSOR_DOES_NOT_EXIST (int): -
          INVALID_PROCESSOR_FOR_JOB (int): -
          RESERVATION_REQUIRED (int): -
        """
        CODE_UNSPECIFIED = 0
        INTERNAL = 1
        INVALID_ARGUMENT = 2
        PERMISSION_DENIED = 3
        PROGRAM_ALREADY_EXISTS = 4
        JOB_ALREADY_EXISTS = 5
        PROGRAM_DOES_NOT_EXIST = 6
        JOB_DOES_NOT_EXIST = 7
        PROCESSOR_DOES_NOT_EXIST = 8
        INVALID_PROCESSOR_FOR_JOB = 9
        RESERVATION_REQUIRED = 10
