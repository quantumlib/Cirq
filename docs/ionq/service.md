# IonQ API

IonQ's API provides a way to execute quantum circuits on IonQ's trapped ion quantum computers
or on cloud based simulators.  As of January 2021 this access is restricted to partners.
See [Access and Authentication](access.md) for details of access.

## Service class

The main entrance for accessing IonQ's API are instances of the `cirq.ionq.Service` class.
These objects can need to be initialized with the remote host url and api key, see [Access and Authentication](access.md) for details.

The basic flow of running a quantum circuit is
1. Create a circuit to run.
1. Create a `cirq.ionq.Service` with proper authentication and endpoints.
3. Submit this circuit to run on the service.
4. Await the results of the service, or alternatively if step 3 was submit more circuits or process
previously run circuits.
5. Transform the results in a form that is most useful for your analysis.

Here is a simple example of this flow
```python
import cirq
import cirq.ionq as ionq

# A circuit that applies a square root of NOT and then a measurement.
qubit = cirq.LineQubit(0)
circuit = cirq.Circuit(
    cirq.X(qubit)**0.5,                # Square root of NOT.
    cirq.measure(qubit, key='result')   # Measurement.
)

# Create a ionq.Service object.
# Replace REMOTE_HOST and API_KEY with your values.
service = ionq.Service(remote_host=REMOTE_HOST, api_key=API_KEY)

# Run a program against the service.
result = service.run(circuit=circuit)

# The return object of run is a cirq.Result object.

# From this object one can get a histogram of results.
print(results.histogram(key='resutl'))

# Or the data as a pandas frame.
print(results.data)
```


bugs

x**0.5 on qpu failed
result object not printable
