## Simulation

Cirq comes with a built in Python simulator for testing out
small circuits.  This simulator can shard its simulation 
across different processes/threads and so take advantage of
multiple cores/CPUs.  

Currently the simulator is tailored to the gate set from
Google's Xmon architecture, but the simulator can be used
to run general gate sets, assuming that you provide an
implementation in terms of this basic gate set.

Here is a simple circuit
```python
import cirq
from cirq import Circuit
from cirq.devices import GridQubit
from cirq.google import ExpWGate, Exp11Gate, XmonMeasurementGate

q0 = GridQubit(0, 0)
q1 = GridQubit(1, 0)

def basic_circuit(meas=True):
    sqrt_x = ExpWGate(half_turns=0.5, axis_half_turns=0.0)
    cz = Exp11Gate()
    yield sqrt_x(q0), sqrt_x(q1)
    yield cz(q0, q1)
    yield sqrt_x(q0), sqrt_x(q1)
    if meas:
        yield XmonMeasurementGate(key='q0')(q0), XmonMeasurementGate(key='q1')(q1)
   
circuit = Circuit()
circuit.append(basic_circuit())
  
print(circuit)
# prints
# (0, 0): ───X^0.5───@───X^0.5───M───
#                    │
# (1, 0): ───X^0.5───@───X^0.5───M───
```

We can simulate this by creating a ``cirq.google.Simulator`` and 
passing the circuit into its ``run`` method:
```python
from cirq.google import XmonSimulator
simulator = XmonSimulator()
result = simulator.run(circuit)

print(result)
# prints something like
# q0=1 q1=1
```
Run returns an ``TrialResult``.  As you can see the result 
contains the result of any measurements for the simulation run.

The actual measurement results here depend on the seeding 
``numpy``s random seed generator. (You can set this using 
``numpy.random_seed``) Another run, can result in a different
set of measurement results:
```python
result = simulator.run(circuit)

print(result)
# prints something like
# q0=1 q1=0
```

The simulator is designed to mimic what running a program
on a quantum computer is actually like.  In particular the 
``run`` methods (``run`` and ``run_sweep``) on the simulator
do not give access to the wave function of the quantum computer
(since one doesn't have access to this on the actual quantum
hardware).  Instead the ``simulate`` methods (``simulate``, 
``simulate_sweep``, ``simulate_moment_steps``) should be used
if one wants to debug the circuit and get access to the full
wave function:
```python
import numpy as np
circuit = Circuit()
circuit.append(basic_circuit(False))    
result = simulator.simulate(circuit, qubit_order=[q0, q1])

print(np.around(result.final_state, 3))
# prints
# [-0.5-0.j   0. -0.5j  0. -0.5j -0.5+0.j ]
```

Note that the simulator uses numpy's ``float32`` precision
(which is ``complex64`` for complex numbers).

### Qubit and Amplitude Ordering

The `qubit_order` argument to the simulator's `run` method 
determines the ordering of some results, such as the 
amplitudes in the final wave function. The `qubit_order`
argument is optional. When it is omitted, qubits are ordered
ascending by their name (i.e. what their `__str__` method returns).

The simplest `qubit_order` value you can provide is a list of 
the qubits in the desired ordered. Any qubits from the circuit 
that are not in the list will be ordered using the 
default `__str__` ordering, but come after qubits that are in
the list. Be aware that all qubits in the list are included in
the simulation, even if they are not operated on by the circuit.

The mapping from the order of the qubits to the order of the 
amplitudes in the wave function can be tricky to understand. 
Basically, it is the same as the ordering used by `numpy.kron`:

```python
outside = [1, 10]
inside = [1, 2]
print(np.kron(outside, inside))
# prints
# [ 1  2 10 20]
```

More concretely, the `k`'th amplitude in the wave function
will correspond to the `k`'th case that would be encountered 
when nesting loops over the possible values of each qubit.
The first qubit's computational basis values are looped over
in the outer-most loop, the last qubit's computational basis 
values are looped over in the inner-most loop, etc:

```python
i = 0
for first in [0, 1]:
    for second in [0, 1]:
        print('amps[{}] is for first={}, second={}'.format(i, first, second))
        i += 1
# prints
# amps[0] is for first=0, second=0
# amps[1] is for first=0, second=1
# amps[2] is for first=1, second=0
# amps[3] is for first=1, second=1
```

We can check that this is in fact the ordering with a 
circuit that flips one qubit out of two:

```python
q_stay = cirq.NamedQubit('q_stay')
q_flip = cirq.NamedQubit('q_flip')
c = cirq.Circuit.from_ops(cirq.X(q_flip))

# first qubit in order flipped
result = simulator.simulate(c, qubit_order=[q_flip, q_stay])
print(abs(result.final_state).round(3))
# prints
# [0. 0. 1. 0.]

# second qubit in order flipped
result = simulator.simulate(c, qubit_order=[q_stay, q_flip])
print(abs(result.final_state).round(3))
# prints
# [0. 1. 0. 0.]
```

### Stepping through a Circuit

Often when debugging it is useful to not just see the end
result of a circuit, but to inspect, or even modify, the 
state of the system at different steps in the circuit.  To
support this Cirq provides a method to return an iterator
over a ``Moment`` by ``Moment`` simulation.  This is the method
``simulate_moment_steps``:
```python
circuit = Circuit()
circuit.append(basic_circuit())    
for i, step in enumerate(simulator.simulate_moment_steps(circuit)):
    print('state at step %d: %s' % (i, np.around(step.state(), 3)))
# prints something like
# state at step 0: [ 0.5+0.j   0.0+0.5j  0.0+0.5j -0.5+0.j ]
# state at step 1: [ 0.5+0.j   0.0+0.5j  0.0+0.5j  0.5+0.j ]
# state at step 2: [-0.5-0.j  -0.0+0.5j -0.0+0.5j -0.5+0.j ]
# state at step 3: [ 0.+0.j  0.+0.j -0.+1.j  0.+0.j]
```

The object returned by the ``moment_steps`` iterator is a 
``XmonStepResult``. This object has the state along with any 
measurements that occurred **during** that step (so does
not include measurement results from previous ``Moments``).
In addition, the``XmonStepResult`` contains ``set_state()``
which  can be used to set the ``state``. One can pass a valid 
full state to this method by passing a numpy array. Or 
alternatively one can pass an integer and then the state
will be set lie entirely in the computation basis state
for the binary expansion of the passed integer.

### Gate sets

The xmon simulator is designed to work with operations that are either a ``GateOperation`` applying an ``XmonGate``,
a ``CompositeOperation`` that decomposes (recursively) to ``XmonGates``,
or a 1-qubit or 2-qubit operation with a ``KnownMatrix``.
By default the xmon simulator uses an ``Extension`` defined in ``xgate_gate_extensions`` to try to resolve gates that are not ``XmonGates`` to ``XmonGates``.

So if you are using a custom gate, there are multiple options
for getting it to work with the simulator:
* Define it directly as an ``XmonGate``.
* Provide a ``CompositeGate`` made up of ``XmonGates``.
* Supply an ``Exentension`` to the simulator which converts
the gate to an ``XmonGate`` or to a ``CompositeGate`` which 
itself can be decomposed in ``XmonGates``. 

### Parameterized Values and Studies

In addition to circuit gates with fixed values, Cirq also 
supports gates which can have ``Symbol`` value (see
[Gates](gates.md)). These are values that can be resolved
at *run-time*. For simulators these values are resolved by
providing a ``ParamResolver``.  A ``ParamResolver`` provides
a map from the ``Symbol``'s name to its assigned value.

```python
from cirq import Symbol, ParamResolver
val = Symbol('x')
rot_w_gate = ExpWGate(half_turns=val)
circuit = Circuit()
circuit.append([rot_w_gate(q0), rot_w_gate(q1)])
for y in range(5):
    resolver = ParamResolver({'x': y / 4.0})
    result = simulator.simulate(circuit, resolver)
    print(np.around(result.final_state, 2))
# prints
# [1.+0.j 0.+0.j 0.+0.j 0.+0.j]
# [ 0.85+0.j    0.  -0.35j  0.  -0.35j -0.15+0.j  ]
# [ 0.5+0.j   0. -0.5j  0. -0.5j -0.5+0.j ]
# [ 0.15+0.j    0.  -0.35j  0.  -0.35j -0.85+0.j  ]
# [ 0.+0.j  0.-0.j  0.-0.j -1.+0.j]
```
Here we see that the ``Symbol`` is used in two gates, and then the resolver
provide this value at run time.

Parameterized values are most useful in defining what we call a
``Study``.  A ``Study`` is a collection of trials, where each 
trial is a run with a particular set of configurations and which
may be run repeatedly.  Running a study returns one 
``TrialContext`` and ``TrialResult`` per set of fixed parameter
values and repetitions (which are reported as the ``repetition_id``
in the ``TrialContext`` object).  Example:
```python
resolvers = [ParamResolver({'x': y / 2.0}) for y in range(3)]
circuit = Circuit()
circuit.append([rot_w_gate(q0), rot_w_gate(q1)])
circuit.append([XmonMeasurementGate(key='q0')(q0), XmonMeasurementGate(key='q1')(q1)])
results = simulator.run_sweep(program=circuit,
                              params=resolvers,
                              repetitions=2)
for result in results:
    print(result)
# prints something like
# repetition_id=0 x=0.0 q0=0 q1=0
# repetition_id=1 x=0.0 q0=0 q1=0
# repetition_id=0 x=0.5 q0=0 q1=1
# repetition_id=1 x=0.5 q0=1 q1=1
# repetition_id=0 x=1.0 q0=1 q1=1
# repetition_id=1 x=1.0 q0=1 q1=1
```
where we see that different repetitions for the case that the
qubit has been rotated into a superposition over computational
basis states yield different measurement results per run.
Also note that we now see the use of the ``TrialContext`` returned as the first
tuple from ``run``: it contains the ``param_dict`` describing what values were
actually used in resolving the ``Symbol``s.

TODO(dabacon): Describe the iterable of parameterized resolvers
supported by Google's API. 
  
### Simulation Configurations and Options

The xmon simulator also contain some extra configuration
on the simulate commands. One of these is ``initial_state``.  
This can be passed the full wave function as a numpy array, or 
the initial state as the binary expansion of a supplied integer 
(following the order supplied by the qubits list). 

A simulator itself can also be passed ``Options`` in it's constructor.
These options define some configuration for how the simulator runs.
For the xmon simulator, these include

> **num_shards**: The simulator works by sharding the wave function
over this many shards. If this is not a power of two, the 
smallest power of two less than or equal to this number will
be used. The sharding shards on the first log base
2 of this number qubit's state. When this is not set the 
simulator will use the number of cpus, which tends to max
out the benefit of multi-processing.

> **min_qubits_before_shard**: Sharding and multiprocessing does
not really help for very few number of qubits, and in fact can
hurt because processes have a fixed (large) cost in Python.
This is the minimum number of qubits that are needed before the
simulator starts to do sharding. By default this is 10.
