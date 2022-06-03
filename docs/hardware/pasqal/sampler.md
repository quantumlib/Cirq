# Pasqal Sampler

The Pasqal sampler, via the `cirq_pasqal.PasqalSampler` class, executes programs and jobs using the
API of Pasqal.

Note that the API of Pasqal is not yet open for public access. Please contact us if you are
interested.

## PasqalSampler class

The `PasqalSampler` class is the entry point to communicate with the API. It can be initialized
using your PASQAL_API_ACCESS_TOKEN.


```python
import cirq
from cirq_pasqal import ThreeDQubit, PasqalVirtualDevice, PasqalSampler

# A simple sample circuit
qubit = ThreeDQubit(0, 0, 0)
p_device = PasqalVirtualDevice(control_radius=2.1, qubits=[qubit])
p_circuit = cirq.Circuit(device=p_device)
p_circuit.append(cirq.X(qubit))                        # NOT gate.
p_circuit.append(cirq.measure(qubit, key='result'))    # Measurement.


# Create a PasqalSampler object to use.
# Replace 'my_token' with the access token and uncomment next lines.

# PASQAL_API_ACCESS_TOKEN = 'my_token'
# sampler = cirq_pasqal.PasqalSampler(remote_host='http://34.98.71.118/v0/pasqal', access_token=PASQAL_API_ACCESS_TOKEN)
# results = sampler.run(p_circuit, repetitions=1000) # Runs the circuit and returns the results in a 'Result'
```

## Device Specification

Currently, only virtual devices are available. The options are:
 * `PasqalVirtualDevice` to emulate a first-generation Pasqal device
 * `PasqalDevice` to work with an unconstrained device which is then optimized and transpiled on Pasqal's side. 
 
 See the [Devices page](devices.md) for more information on how to work with these devices.

## Calibration Metrics

Not yet available.
