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
"""Helper for testing python logging statements."""

import logging
from typing import ContextManager, List, Optional


def assert_logs(
    *matches: str,
    count: Optional[int] = 1,
    min_level: int = logging.WARNING,
    max_level: int = logging.CRITICAL,
    capture_warnings: bool = True,
) -> ContextManager[List[logging.LogRecord]]:
    """A context manager for testing logging and warning events.

    To use this one wraps the code that is to be tested for log events within
    the context of this method's return value:

        with assert_logs(count=2, 'first_match', 'second_match') as logs:
            <code that produces python logs>

    This captures the logging that occurs in the context of the with statement,
    asserts that the number of logs is equal to `count`, and then asserts that
    all of the strings in `matches` appear in the messages of the logs.
    Further, the captured logs are accessible as `logs` and further testing
    can be done beyond these simple asserts.

    Args:
        matches: Each of these is checked to see if they match, as a substring,
            any of the captures log messages.
        count: The expected number of messages in logs. Defaults to 1. If None is passed in counts
            are not checked.
        min_level: The minimum level at which to capture the logs. See the python logging
            module for valid levels. By default this captures at the
            `logging.WARNING` level and above, so this does not capture `logging.INFO`
            or `logging.DEBUG` logs by default.
        max_level: The maxium level at which to capture the logs. See the python logging
            module for valid levels. By default this captures to the `logging.CRITICAL` level
            thus, all the errors and critical messages will be captured as well.
        capture_warnings: Whether warnings from the python's `warnings` module
            are redirected to the logging system and captured.

    Returns:
        A ContextManager that can be entered into which captures the logs
        for code executed within the entered context. This ContextManager
        checks that the asserts for the logs are true on exit.

    Raises:
        ValueError: If `min_level` is greater than `max_level`.
    """
    if min_level > max_level:
        raise ValueError("min_level should be less than or equal to max_level")
    records = []

    class Handler(logging.Handler):
        def emit(self, record):
            # filter only the interesting ones
            if max_level >= record.levelno >= min_level:
                records.append(record)

        def __enter__(self):
            logging.captureWarnings(capture_warnings)
            logger = logging.getLogger()
            # we capture all the logs
            logger.setLevel(logging.DEBUG)
            logger.addHandler(self)
            return records

        def __exit__(self, exc_type, exc_val, exc_tb):
            logging.getLogger().removeHandler(self)
            if capture_warnings:
                logging.captureWarnings(False)
            msgs = [record.getMessage() for record in records]

            assert count is None or len(records) == count, (
                f'Expected {count} log message but ' f'got {len(records)}. Log messages: ' f'{msgs}'
            )
            for match in matches:
                assert match in ''.join(msgs), (
                    f'{match} expected to appear in log messages but it was '
                    f'not found. Log messages: {msgs}.'
                )

    return Handler()


# pylint: enable=missing-raises-doc
