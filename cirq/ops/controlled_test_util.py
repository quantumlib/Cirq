from typing import Union, Tuple, cast

import cirq
import numpy as np
from cirq.type_workarounds import NotImplementedType


class GateUsingWorkspaceForApplyUnitary(cirq.SingleQubitGate):
    def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs
                        ) -> Union[np.ndarray, NotImplementedType]:
        args.available_buffer[...] = args.target_tensor
        args.target_tensor[...] = 0
        return args.available_buffer

    def _unitary_(self):
        return np.eye(2)

    def __pow__(self, exponent):
        return self

    def __eq__(self, other):
        return isinstance(other, type(self))

    def __repr__(self):
        return ('cirq.ops.controlled_gate_test.'
                'GateUsingWorkspaceForApplyUnitary()')


class GateAllocatingNewSpaceForResult(cirq.SingleQubitGate):
    def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs
                        ) -> Union[np.ndarray, NotImplementedType]:
        assert len(args.axes) == 1
        a = args.axes[0]
        seed = cast(Tuple[Union[int, slice, 'ellipsis'], ...],
                    (slice(None),))
        zero = seed*a + (0, Ellipsis)
        one = seed*a + (1, Ellipsis)
        result = np.zeros(args.target_tensor.shape, args.target_tensor.dtype)
        result[zero] = args.target_tensor[zero]*2 + args.target_tensor[one]*3
        result[one] = args.target_tensor[zero]*5 + args.target_tensor[one]*7
        return result

    def _unitary_(self):
        return np.array([[2, 3], [5, 7]])

    def __pow__(self, factor):
        return self

    def __eq__(self, other):
        return isinstance(other, type(self))

    def __repr__(self):
        return ('cirq.ops.controlled_gate_test.'
                'GateAllocatingNewSpaceForResult()')
