## Nomenclature

### Quantum computing terms

Use `state_vector` to describe a pure state that is passed in as a numpy
array, numpy tensor, or list of amplitudes. **Do not** use `wavefunction`, 
`wave_function`, or `state` for this object (`state` is too overloaded).  
If the object is  an array or possibly a computational basis state 
(given by an `int`), use `state_rep`.

Use `density_matrix` to describe a mixed state that is passed in as a numpy
matrix or numpy tensor.  **Do not** used `mixed_state`, `density_operator`, or
`state`.