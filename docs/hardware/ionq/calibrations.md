# IonQ API Calibrations

Calibrations are snapshots of the performance of the IonQ platform
at a given moment of time.  The IonQ quantum computers are continuously
calibrated, and measurements are taken of the performance of the machine
periodically.

In this section, we assume a `cirq_ionq.Service` object has been instantiated and is
called `service` and `cirq` and `cirq_ionq` have been imported:
```python
import cirq
import cirq_ionq as ionq
service = ionq.Service()
```
See [IonQ API Service](service.md) for how to set up the service.

## Fetching information about the device

To get the latest calibration, you simply query the service for the calibration
object
```python
calibration = service.get_current_calibration()
```
The returned object then has data about the target (currently only on QPU)
as well as the current performance of the target.

```python
print(calibration.fidelities())
# prints something like
{'1q': {'mean': 0.99717}, '2q': {'mean': 0.9696}, 'spam': {'mean': 0.9961}}
```
Here
* `1q` and `2q` refer to one and two qubit average gate fidelities measured using
randomized benchmarking.  Technically these refer to the native gates on the
device, which are the `cirq.XX**(0.5)` gate and the `cirq.X**(0.5)` gate (and also their
inverses).  Like many architectures `cirq.Z**x` gates are "free", these gates are
compiled out of the circuit.  Thus these gates are not included in
this gate fidelities.
* `spam` here refers to state preparation and measurement
error, and can mostly be thought of as the probability of the measurement
being correct.

Another useful bit of information are the timings of the gates:
```python
print(calibration.timings())
# prints something like
{'t1': 10000, 't2': 0.2, '1q': 1.1e-05, '2q': 0.00021, 'readout': 0.000175, 'reset': 3.5e-05}
```
These measurements are all returned in seconds
* `t1`: the energy relaxation time.
* `t2`: the dephasing time.
* `1q`: the time it takes to execute a single qubit gate.
* `2q`: the time it takes to execute a two qubit gate.
* `readout`: the time it takes to measure the qubit.
* `reset`: the time it takes to reset the ion between shots.

A few other properties of the `ionq.Calibration` object are
* `num_qubits`: the number of qubits on the QPU.
* `connectivity`: a set of all the possible qubits that can interact in the set.
* `target`: in the future when there are multiple QPUs this will list which target.
