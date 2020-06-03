# # Copyright 2019 The Cirq Developers
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     https://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# """A protocol for implementing high performance clifford tableau evolutions for Clifford Simulator."""
#
# from typing import Any, cast, Iterable, Optional, Tuple, TypeVar, Union
#
# from cirq.ops.global_phase_op import GlobalPhaseOperation
# from cirq.ops.raw_types import Operation
# from cirq.sim.clifford.clifford_simulator import CliffordState
#
# # This is a special indicator value used by the apply_clifford_tableau method
# # to determine whether or not the caller provided a 'default' argument. It must
# # be of type CliffordState to ensure the method has the correct type signature
# # in that case. It is checked for using `is`, so it won't have a false positive
# # if the user provides a different CliffordState value.
#
# RaiseTypeErrorIfNotProvided = CliffordState([])  # type: cirq.CliffordState
#
# TDefault = TypeVar('TDefault')
#
# def apply_clifford_tableau(val: Operation, state: 'cirq.CliffordState',
#                   default: TDefault = RaiseTypeErrorIfNotProvided
#                           ) -> Union['cirq.CliffordState', TDefault]:
#     """
#
#     :param qubits:
#     :param val:
#     :param state:
#     :return:
#     """
#     if state is None:
#         raise ValueError('Input CliffordState cannot be empty.')
#     strats = [
#         _strat_handle_global_phase_op, _strat_apply_tableau_by_magic_method, _strat_handle_single_qubit_unitary
#     ]
#     for strat in strats:
#         result = strat(val, state)
#         if result is None:
#             break
#         if result is not NotImplemented:
#             return result
#
#     # Don't know how to apply. Fallback to specified default behavior.
#     if default is not RaiseTypeErrorIfNotProvided:
#         return default
#     raise ValueError(
#         "cirq.apply_clifford_tableau failed. "
#         "Operation doesn't have a (non-parameterized) clifford effect.\n"
#         "\n"
#         "type: {}\n"
#         "value: {!r}\n"
#         "\n"
#         "The Operation failed to satisfy any of the following criteria:\n"
#         "- Is a `GlobalPhaseOperation`.\n"
#         "- An `_apply_clifford_tableau_(self, state, qubits) method on the "
#         "Gate that returned a value besides None or NotImplemented.\n"
#         "- Is a single qubit operation that can be decomposed into Clifford "
#         "rotations.\n"
#         "".format(type(val), val))
#
#
# def _strat_handle_global_phase_op(val: Operation, state: 'cirq.CliffordState'
#                                  ) -> Optional['cirq.CliffordState']:
#     if isinstance(val, GlobalPhaseOperation):
#         state.ch_form.omega *= val.coefficient
#         return state
#     return NotImplemented
#
#
# def _strat_apply_tableau_by_magic_method(val: Operation,
#                                          state: 'cirq.CliffordState'
#                                         ) -> Optional['cirq.CliffordState']:
#     getter = getattr(val.gate, '_apply_clifford_tableau_', None)
#     if getter is not None:
#         return getter(state, val.qubits)
#     return NotImplemented
#
# def _strat_handle_single_qubit_unitary(val: Operation,
#                                          state: 'cirq.CliffordState'
#                                         ) -> Optional['cirq.CliffordState']:
#     if len(val.qubits) != 1:
#         return NotImplemented
#     state.apply_single_qubit_unitary(val)
#     return state