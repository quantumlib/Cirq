## What does this PR do?
This PR introduces `DropObservableIrrelevantOps`, a new transformer designed to safely remove operations that are provably irrelevant to the observable measurement outcomes of a circuit. 

Unlike simple peephole optimizations, this pass implements a strict, time-aware, and observable-context-aware static analysis. It evaluates operations against the temporal boundaries defined by independent measurement keys, ensuring zero false positives and strict adherence to classical register semantics.

## Motivation
Currently, removing diagonal gates (like Z or CZ) before measurements requires specific pattern matching (e.g., `drop_diagonal_before_measurement`). However, when dealing with complex circuits featuring multiple independent observers (different measurement keys) or temporally separated measurements, a more robust, generalized approach is needed. 

This transformer strictly isolates measurement keys as independent observable contexts and uses moment indices to establish definitive causal boundaries.

## Implementation Details
The optimization strictly follows this existential quantification:
An operation `op` at `moment_index` is removable if and only if **$\exists$ key** such that:
1. `moment_index < first_measurement_moment[key]` (Strict temporal causality)
2. `op.qubits ⊆ measured_qubits[key]` (Strict spatial observability)
3. `is_diagonal_in_z(op)` (Strict phase-only effect)

### Key Invariants Guaranteed:
* **Observer Independence:** Different measurement keys are treated as completely independent contexts. An operation irrelevant to one observer can be safely removed even if another observer exists elsewhere in the circuit.
* **Temporal Monotonicity:** For any given measurement key, the *first* occurrence strictly defines the event horizon. Subsequent measurements (even on a subset of qubits) do not extend the removal window, strictly preserving classical register semantics and preventing false positives from phase kickback.
* **Conservative Safety:** The `is_diagonal_in_z` check is explicitly whitelisted (Z, CZ, ZPowGate) to prevent over-optimization of unknown gate types.

## Testing Strategy
Comprehensive test coverage is included, proving the invariants:
- [x] Basic removal before Z-basis measurements.
- [x] Preservation of causal ordering (no removal after measurement).
- [x] Preservation during partial/subset measurements (entanglement safety).
- [x] Independence of multiple measurement keys (no cross-interference).
- [x] Temporal crossing resolution (operations between two different key measurements).
- [x] Strict boundary enforcement on subset remeasurements for the same key.

## Checklist
- [x] Tests pass
- [x] Docstrings are complete and formal
- [x] Pure functional transformation (no side effects)
