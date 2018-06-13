from typing import Dict, Any, Iterable

import numpy as np
from cirq import linalg


class SimulateCircuitResult:
    """Measurements and final state vector from a circuit simulation.

    Attributes:
        measurements: A dictionary from measurement gate key to measurement
            results. Each measurement result value is a numpy ndarray of actual
            boolean measurement results (ordered by the qubits acted on by the
            measurement gate.)
        final_state: The state vector output by the circuit. The final state
            of the simulated system.
    """

    def __init__(self,
                 measurements: Dict[str, np.ndarray],
                 final_state: np.ndarray) -> None:
        """
        Args:
            measurements: A dictionary from measurement gate key to measurement
                results. Each measurement result value is a numpy ndarray of
                actual boolean measurement results (ordered by the qubits acted
                on by the measurement gate.)
            final_state: The state vector output by the circuit. The final
                state of the simulated system.
        """
        self.measurements = measurements
        self.final_state = final_state

    def __repr__(self):
        return ('SimulateCircuitResult(measurements={!r}, '
                'final_state={!r})').format(self.measurements,
                                            self.final_state)

    def __str__(self):
        return 'measurements: {}\nfinal_state: {}'.format(
            _keyed_iterable_bitstrings(self.measurements),
            self.final_state)

    def approx_eq(self,
                  other: 'SimulateCircuitResult',
                  atol: float=0,
                  ignore_global_phase=True) -> bool:
        if len(self.measurements) != len(other.measurements):
            return False

        for k, v in self.measurements.items():
            other_v = other.measurements.get(k)
            if (other_v is None or
                    len(other_v) != len(v) or
                    np.any(v != other_v)):
                return False

        cmp_vector = (linalg.allclose_up_to_global_phase if ignore_global_phase
                      else np.allclose)
        return cmp_vector(self.final_state, other.final_state, atol=atol)


def _iterable_bitstring(vals: Iterable[Any]) -> str:
    return ''.join('1' if v else '0' for v in vals)


def _keyed_iterable_bitstrings(keyed_vals: Dict[str, Iterable[Any]]) -> str:
    if not keyed_vals:
        return '(none)'
    results = sorted(
        (key, _iterable_bitstring(val)) for key, val in keyed_vals.items())
    return ' '.join(['{}={}'.format(key, val) for key, val in results])
