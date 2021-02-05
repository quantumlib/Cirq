# Calibrations

Calibrations are snapshots of the performance of the IonQ platform
at a given moment of time.  The IonQ quantum computers are continuously
calibrated, and we periodically take measurements of the performance
of the machine.

In this section we assume a `cirq.ionq.Service` object has been instantiated and is
called `service` and `cirq` and `cirq.ionq` have been imported:
```python
import cirq
import cirq.ionq as ionq
service = ionq.Service()
```
See [IonQ API Service](service.md) for how to set the service up.

## Fetching information about the device

To get the latest calibration, one simply queries the service for the calibration
object
```python
calibration = service.get_current_calibration()
```
The returned object then has data both about the target (currently only on QPU)
as well as the current performance of the target.

```python
print(calibration.fidelities())
# prints something like
{'1q': {'mean': 0.99717}, '2q': {'mean': 0.9696}, 'spam': {'mean': 0.9961}}
>>>
```
Here
* `1q` and `2q` refer to one and two qubit average gate fidelities measured using
randomized benchmarking.  Technically these refer to the native gates on the
device, which are the `cirq.XX**(\pm 0.5)` gate and the `cirq.X**(\pm 0.5)` gate.  Like
many architectures `cirq.Z**\alpha` gates are "free": these gates are compiled out of the
circuit, changing the single qubit gates and being absorbed into the two qubit gates
when necessary.
* `spam` here refers to state preparation and measurement
error, and can mostly be thought of as the probability of the measurement
being correct.

Another useful bit of information are the timings of the gates:
```python
print(calibration.timings())
# prints something like
{'t1': 10000, 't2': 0.2, '1q': 1.1e-05, '2q': 0.00021, 'readout': 0.000175, 'reset': 3.5e-05}
```
These measurements are all returned in second
* `t1`: the energy relaxation time.
* `t2`: the dephasing time.
* `1q`: the time it takes to implement a single qubit gate.
* `2q`: the time it takes to implement a two qubit gate.
* `readout`: the time it takes to measure the qubit.
* `reset`: the time it takes to reset the ion between shots.

A few other properties of the `ionq.Calibration` object are
* `num_qubits`: the number of qubits on the QPU.
* `connectivity`: a set of all the possible qubits that can interact in the set.
* `target`: in the future when there are multiple QPUs this will list which target.
