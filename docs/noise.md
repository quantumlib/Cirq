## Noise

For simulation it is useful to have `Gate`s that enact noisy quantum evolution.
Currently Cirq supports modeling noise via *operator sum* representations of
noise (these evolutions are also known as quantum operations or quantum
dynamical maps).  This formalism models evolution of the density matrix
via

![Operator sum representation: $\rho \rightarrow \sum_k A_k \rho A_k^\dagger$](resources/OperatorSumDef.gif)

Where here the the A<sub>k</sub> are *Krauss* operators. These operators
are not necessarily unitary and must satisfy the trace preserving
property

![Operator sum normalization: $\sum_k A_k^\dagger A_k = I$](resources/OperatorSumNormDef.gif)

### Magic methods

A `Gate` can represent a operator sum representation by supporting the
`channel` protocol.

#### cirq.channel and def _channel_


#### cirq.has_channel and def _has_channel_


### Common Channels