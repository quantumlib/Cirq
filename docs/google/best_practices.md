# Best Practices

This section lists some best practices for creating a circuit that performs well
on Google hardware devices. This is an area of active research, so users are
encouraged to try multiple approaches to improve results.

## Use built-in optimizers as a first pass

Using built-in optimizers will allow you to compile to the correct gate set. As they are
automated solutions, they will not always perform as well as a hand-crafted solution, but
they provide a good starting point for creating a circuit that is likely to run successfully
on hardware. Best practice is to inspect the circuit after optimization to make sure
that it has compiled without unintended consequences.

```python
import cirq
import cirq.google as cg


# Create your circuit here
my_circuit = cirq.Circuit()

# Convert the circuit onto a Google device.
# Specifying a device will verify that the circuit satisfies constraints of the device
# The optimizer type (e.g. 'sqrt_iswap' or 'sycamore') specifies which gate set
# to convert into and which optimization routines are appropriate.
# This can include combining successive one-qubit gates and ejecting virtual Z gates. 
sycamore_circuit = cg.optimized_for_sycamore(my_circuit, new_device=cg.Sycamore, optimizer_type='sqrt_iswap')
```

## Good moment structure

Quantum Engine will execute a circuit as faithfully as possible.
This means moment structure will be preserved. That is, all gates in a moment are
guaranteed to be executed before those in any later moment and after gates in
previous moments. 

To this end, it is important that the moment structure of the circuit is kept as
short and concise as possible. The length of a moment will generally be the length
of the longest gate in the moment, so keeping gates with similar durations together
will shorten the duration of the circuit and likely reduce the noise incurred.

In particular, keep measurement gates in the same moment and make sure that any
circuit optimizers do not alter this by pushing measurements forward. This
behavior can be avoided by measuring all qubits with a single gate or by adding
the measurement gate after all optimizers have run.

## Short gate depth

In the current NISQ (noisy intermediate scale quantum) era, gates and devices still
have significant error. Both gate errors and T1 decay rate can cause long circuits
to have noise that overwhelms any signal in the circuit. 

The recommended gate depths vary significantly with the structure of the circuit itself
and will likely increase as the devices improve. Total circuit fidelity can be roughly
estimated by multiplying the fidelity for all gates in the circuit. For example,
using a error rate of 0.5% per gate, a circuit of depth 20 and width 20 could be estimated
at 0.995^(20*20) = 0.135. Using separate error rates per gates (i.e. based on calibration
metrics) or a more complicated noise model can result in more accurate error estimation.

## Use sweeps when possible

Round trip network time to and from the engine typically adds latency on the order of a second
to the overall computation time.  Reducing the number of trips and allowing the engine to
properly batch circuits can improve the throughput of your calculations.  One way to do this
is to use parameter sweeps to send multiple variations of a circuit at once.

One example is to turn single-qubit gates on or off by using parameter sweeps.  
For instance, the following code illustrates how to combine measuring in the
Z basis or the X basis in one circuit.

```python
import cirq
import sympy
q = cirq.GridQubit(1, 1)
sampler = cirq.Simulator()

# STRATEGY #1: Have a separate circuit and sample call for each basis.
circuit_z = cirq.Circuit(
    cirq.measure(q, key='out'))
circuit_x = cirq.Circuit(
    cirq.H(q),
    cirq.measure(q, key='out'))
samples_z = sampler.sample(circuit_z, repetitions=5)
samples_x = sampler.sample(circuit_x, repetitions=5)

print(samples_z)
# prints
#    out
# 0    0
# 1    0
# 2    0
# 3    0
# 4    0

print(samples_x)
# prints something like:
#    out
# 0    0
# 1    1
# 2    1
# 3    0
# 4    0

# STRATEGY #2: Have a parameterized circuit.
circuit_sweep = cirq.Circuit(
    cirq.H(q)**sympy.Symbol('t'),
    cirq.measure(q, key='out'))

samples_sweep = sampler.sample(circuit_sweep,
                               repetitions=5,
                               params=[{'t': 0}, {'t': 1}])
print(samples_sweep)
# prints something like:
#    t  out
# 0  0    0
# 1  0    0
# 2  0    0
# 3  0    0
# 4  0    0
# 0  1    0
# 1  1    1
# 2  1    1
# 3  1    0
# 4  1    1
```

One word of caution is there is a limit to the total number of repetitions.  Take some care
that your parameter sweeps, especially products of sweeps, do not become so excessively large
that they overcome this limit.


## Keep qubits busy
 
Qubits that remain idle for long periods tend to dephase and decohere. Inserting a
[Spin Echo](https://en.wikipedia.org/wiki/Spin_echo) into your circuit, such as a pair
of involutions, such as two successive Pauli Y gates, will generally increase
performance of the circuit.

## Alternate single-qubit and two-qubit layers

Devices are generally calibrated to circuits that alternate single-qubit gates with
two-qubit gates in each layer. Staying close to this paradigm will often improve
performance of circuits.

Devices generally operate in the Z basis, so that rotations around the Z axis will become
book-keeping measures rather than physical operations on the device. The EjectZ optimizer
included in optimizer lists for each device will generally compile these operations out
of the circuit by pushing them back to the next non-commuting operator. If the resulting
circuit still contains Z operations, they should be aggregated into their own moment,
if possible.

## Use caution with symbols

Symbols are extremely useful for constructing parameterized circuits (see above).  However,
only some sympy formulas can be serialized for network transport to the engine.
Currently, sums and products of symbols, including linear combinations, are supported.
See `cirq.google.arg_func_langs` for details.

The sympy library is also infamous for being slow, so avoid using complicated formulas if you
care about performance.  Avoid using parameter resolvers that have formulas in them. 

One way to eliminate formulas in your gates is to flatten your expressions.
The following example shows how to take a gate with a formula and flatten it
to a single symbol with the formula pre-computed for each value of the sweep:

```python
import cirq
import sympy

# Suppose we have a gate with a complicated formula.  (e.g. "2^t - 1")
# This formula cannot be serialized
# It could potentially encounter sympy slowness.
gate_with_formula = cirq.XPowGate(exponent=2 ** sympy.Symbol('t') - 1)
sweep = cirq.Linspace('t', start=0, stop=1, length=5)

# Instead of sweeping the formula, we will pre-compute the values of the formula
# at every point and store it a new symbol called '<2**t - 1>'
sweep_for_gate, flat_sweep = cirq.flatten_with_sweep(gate_with_formula, sweep)

print(sweep_for_gate)
# prints:
# (cirq.X**sympy.Symbol('<2**t - 1>'))

# The sweep now contains the non-linear progression of the formula instead:
print(list(flat_sweep.param_tuples()))
# prints something like:
# [(('<2**t - 1>', 0.0),),
#  (('<2**t - 1>', 0.18920711500272103),),
#  (('<2**t - 1>', 0.41421356237309515),),
#  (('<2**t - 1>', 0.681792830507429),),
#  (('<2**t - 1>', 1.0),)]
```