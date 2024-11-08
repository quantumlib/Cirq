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

from typing import Any, Dict, Callable, Optional, Sequence
from collections.abc import Mapping

import numpy as np

from cirq import ops, value
from cirq_google.api.v2 import program_pb2


SUPPORTED_INTERPOLATION_METHODS = frozenset(
    [
        'interp',  # np.interp
        'CubicSpline',  # scipy.interpolate.CubicSpline
        'PchipInterpolator',  # scipy.interpolate.PchipInterpolator
        'Akima1DInterpolator',  # scipy.interpolate.Akima1DInterpolator
    ]
)


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
        custom_args: Mapping[program_pb2.CustomArg, Any] = None,
        **kwargs,
    ):
        """Instatiates an InternalGate.

        Arguments:
            gate_name: Gate class name.
            gate_module: The module of the gate.
            num_qubits: Number of qubits that the gate acts on.
            **kwargs: The named arguments to be passed to the gate constructor.
        """
        self.gate_module = gate_module
        self.gate_name = gate_name
        self._num_qubits = num_qubits
        self.gate_args = kwargs
        self.custom_args = custom_args

    def _num_qubits_(self) -> int:
        return self._num_qubits

    def __str__(self):
        gate_args = ', '.join(f'{k}={v}' for k, v in self.gate_args.items())
        return f'{self.gate_module}.{self.gate_name}({gate_args})'

    def __repr__(self) -> str:
        gate_args = ', '.join(f'{k}={repr(v)}' for k, v in self.gate_args.items())
        if gate_args != '':
            gate_args = ', ' + gate_args
        return (
            f"cirq_google.InternalGate(gate_name='{self.gate_name}', "
            f"gate_module='{self.gate_module}', "
            f"num_qubits={self._num_qubits},"
            f"custom_args={self.custom_args}"
            f"{gate_args})"
        )

    def _json_dict_(self) -> Dict[str, Any]:
        return dict(
            gate_name=self.gate_name,
            gate_module=self.gate_module,
            num_qubits=self._num_qubits,
            custom_args=self.custom_args,
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
        )


def encode_1dfunction(
    x: Optional[Sequence[float]] = None,
    y: Optional[Sequence[float]] = None,
    method: str = 'interp',
    *,
    f: Optional[Callable[[float], float]] = None,
    x_low: float = 0,
    x_high: float = 1,
    num_points: int = 100,
) -> program_pb2.CustomArg:
    """Encodes a general 1D-function as a list of evaluations.

    This method discretizes a function into a list of point evaluations. Evaluating the
    function on general points is then done by interpolating the function using the given
    method.

    Args:
        x: Optional list of values of the free variable.
            If not given, then np.linspace(x_low, x_high, num_points) is used.
        y: Optional list of values of the dependent variable.
            If not given, then evaluations of f(x) are used.
        method: The method used to interpolate the function.

        f: A callable to populate `y` if `y` is not given.
        x_low: Smallest value of the free variable. Only used if `x` is not given.
        x_high: Largest value of the free variable. Only used if `x` is not given.
        num_points: Number of points to use if `x` is not given.

    Returns:
        A CustomArg encoding the function.

    Raises:
        ValueError: If
            - neigther `y` nor `y` is given.
            - `method` is not supported.
    """

    if method not in SUPPORTED_INTERPOLATION_METHODS:
        raise ValueError(
            f'Method {method} is not supported. The supported methods are {SUPPORTED_INTERPOLATION_METHODS}'
        )

    if x is None:
        x = np.linspace(x_low, x_high, num_points)

    if y is None:
        if f is None:
            raise ValueError(f'At least one of y and f must be given.')
        y = [f(v) for v in x]

    msg = program_pb2.CustomArg()
    msg.function_interpolation_data.x.extend(x)
    msg.function_interpolation_data.y.extend(y)
    msg.function_interpolation_data.method = method
    return msg
