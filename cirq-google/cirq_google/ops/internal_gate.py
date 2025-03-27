# Copyright 2023 The Cirq Developers
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

from collections.abc import Mapping
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np

from cirq import ops, value
from cirq_google.api.v2 import program_pb2


@value.value_equality
class InternalGate(ops.Gate):
    """InternalGate is a placeholder gate for internal gates.

    InternalGate holds the information required to instantiate
    a gate of type `self.gate_name` with the arguments for the gate
    constructor stored in `self.gate_args`.
    """

    def __init__(
        self,
        gate_name: str,
        gate_module: str,
        num_qubits: int = 1,
        custom_args: Optional[Mapping[str, program_pb2.CustomArg]] = None,
        **kwargs,
    ):
        """Instatiates an InternalGate.

        Arguments:
            gate_name: Gate class name.
            gate_module: The module of the gate.
            num_qubits: Number of qubits that the gate acts on.
            custom_args: A mapping from argument name to `CustomArg`.
                This is to support argument that require special processing.
            **kwargs: The named arguments to be passed to the gate constructor.
        """
        self.gate_module = gate_module
        self.gate_name = gate_name
        self._num_qubits = num_qubits
        self.gate_args = kwargs
        self.custom_args = custom_args or {}

    def _num_qubits_(self) -> int:
        return self._num_qubits

    def __str__(self):
        gate_args = ', '.join(f'{k}={v}' for k, v in (self.gate_args | self.custom_args).items())
        return f'{self.gate_module}.{self.gate_name}({gate_args})'

    def __repr__(self) -> str:
        gate_args = ', '.join(f'{k}={repr(v)}' for k, v in self.gate_args.items())
        if gate_args != '':
            gate_args = ', ' + gate_args

        custom_args = ''
        if self.custom_args:
            custom_args = f", custom_args={self.custom_args}"

        return (
            f"cirq_google.InternalGate(gate_name='{self.gate_name}', "
            f"gate_module='{self.gate_module}', "
            f"num_qubits={self._num_qubits}"
            f"{custom_args}"
            f"{gate_args})"
        )

    def _json_dict_(self) -> Dict[str, Any]:
        if self.custom_args:
            raise ValueError('InternalGate with custom args are not json serializable')
        return dict(
            gate_name=self.gate_name,
            gate_module=self.gate_module,
            num_qubits=self._num_qubits,
            **self.gate_args,
        )

    def _value_equality_values_(self):
        hashable = True
        for arg in self.gate_args.values():
            try:
                hash(arg)
            except TypeError:
                hashable = False
        return (
            self.gate_module,
            self.gate_name,
            self._num_qubits,
            frozenset(self.gate_args.items()) if hashable else self.gate_args,
            frozenset((k, v.SerializeToString()) for k, v in self.custom_args.items()),
        )


def function_points_to_proto(
    x: Union[Sequence[float], np.ndarray],
    y: Union[Sequence[float], np.ndarray],
    msg: Optional[program_pb2.CustomArg] = None,
) -> program_pb2.CustomArg:
    """Return CustomArg that expresses a function through its x and y values.

    Args:
        x: Sequence of values of the free variable.
            For 1D functions, this input is assumed to be given in increasing order.
        y: Sequence of values of the dependent variable.
            Where y[i] = func(x[i]) where `func` is the function being encoded.
        msg: Optional CustomArg to serialize to.
            If not provided a CustomArg is created.

    Returns:
        A CustomArg encoding the function.

    Raises:
        ValueError: If
            - `x` is 1D and not sorted in increasing order.
            - `x` and `y` don't have the same number of points.
            - `y` is multidimensional.
            - `x` is multidimensional.
    """

    x = np.asarray(x)
    y = np.asarray(y)

    if len(x.shape) != 1:
        raise ValueError('The free variable must be one dimensional')

    if len(x.shape) == 1 and not np.all(np.diff(x) > 0):
        raise ValueError('The free variable must be sorted in increasing order')

    if len(y.shape) != 1:
        raise ValueError('The dependent variable must be one dimensional')

    if x.shape[0] != y.shape[0]:
        raise ValueError('Mismatch between number of points in x and y')

    if msg is None:
        msg = program_pb2.CustomArg()
    msg.function_interpolation_data.x_values[:] = x
    msg.function_interpolation_data.y_values[:] = y
    return msg
