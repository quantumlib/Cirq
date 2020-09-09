# Nomenclature

This page describes naming rules and conventions that exist throughout Cirq.
These rules are important in order to maintain a consistent interface that is 
easy to use. By using consistent naming we can reduce cognitive load on 
users and developers. Please try to use these terms when writing code.

If you have suggestions for improvements or changes, please create a PR 
to modify this list or open an issue.

## Quantum computing terms

Use `state_vector` to describe a pure state that is passed in as a numpy
array, numpy tensor, or list of amplitudes. **Do not** use `wavefunction`, 
`wave_function`, or `state` for this object (`state` is too overloaded).  
If the object is an array or possibly a computational basis state 
(given by an `int`), use `state_rep` or, if it is the initial state of 
a system `initial_state`.

Use `density_matrix` to describe a mixed state that is passed in as a numpy
matrix or numpy tensor.  **Do not** used `mixed_state`, `density_operator`, or
`state`.