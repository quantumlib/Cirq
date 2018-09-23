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

"""Hard-coded options for adding multiple operations to a circuit."""


class InsertStrategy(object):
    """Indicates preferences on how to add multiple operations to a circuit."""

    NEW = None  # type: InsertStrategy
    NEW_THEN_INLINE = None  # type: InsertStrategy
    INLINE = None  # type: InsertStrategy
    EARLIEST = None  # type: InsertStrategy

    def __init__(self, name, doc):
        self.name = name
        self.__doc__ = doc

    def __str__(self):
        return self.name

    def __repr__(self):
        return 'cirq.InsertStrategy.{}'.format(self.name)


InsertStrategy.NEW = InsertStrategy(
    'NEW',
    """
    Always creates a new moment at the desired insert location, and adds the
    operation to insert into that moment.
    """)

InsertStrategy.NEW_THEN_INLINE = InsertStrategy(
    'NEW_THEN_INLINE',
    """
    Creates a new moment at the desired insert location for the first
    operation, but then switches to inserting operations inline.
    """)

InsertStrategy.INLINE = InsertStrategy(
    'INLINE',
    """
    Attempts to add the operation to insert into the moment just before the
    desired insert location. But, if there's already an existing operation
    affecting any of the qubits touched by the operation to insert, or the
    desired index is 0, a new moment is created and inserted at the desired
    location instead.
    In case the insert index is smaller than -len(<moments-in-circuit>) it is
    treated like it would be 0.
    For too big indices it attempts to insert the operation into the last
    moment of the circuit.
    """)

InsertStrategy.EARLIEST = InsertStrategy(
    'EARLIEST',
    """
    Scans backward from the insert location until a moment with operations
    touching qubits affected by the operation to insert is found. The operation
    is added into the moment just after that location.

    If the scan reaches the start of the circuit without finding any conflicting
    operations, the operation is added into the first moment of the circuit.

    The operation is never added into moments after the insert location.
    If the moment just before the insert location has conflicting operations,
    or the insert index is 0, then the operation is inserted into a new moment
    at the desired location.
    """)
