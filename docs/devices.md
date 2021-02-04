# Devices

``Device`` is an abstract concept in Cirq, that can represent constraints of an actual quantum processor. 
This page describes this abstract concept. 

If you are looking for ways of running quantum algorithms, take a look at 
 - [Simulation](simulation.ipynb), that is available on any computer
 - Quantum processors, that are provided by different Quantum Service Providers: 
    - [Google Quantum Computing Service](tutorials/google/start.ipynb)
    - [Alpine Quantum Technologies](tutorials/aqt/getting_started.ipynb)
    - [Pasqal](tutorials/pasqal/getting_started.ipynb) 

## The `cirq.Device` class

The ``Device`` class is an abstract class which encapsulates constraints
(or lack thereof) that come when running a circuit on actual hardware.
For instance, most hardware only allows certain gates to be enacted
on qubits.  Or, as another example, some gates may be constrained to not
be able to run at the same time as neighboring gates.  Further the
``Device`` class knows more about the scheduling of ``Operations``.

Here for example is a ``Device`` made up of 10 qubits on a line:
```python
import cirq
from cirq.devices import GridQubit
class Xmon10Device(cirq.Device):

  def __init__(self):
      self.qubits = [GridQubit(i, 0) for i in range(10)]

  def validate_operation(self, operation):
      if not isinstance(operation, cirq.GateOperation):
          raise ValueError('{!r} is not a supported operation'.format(operation))
      if not isinstance(operation.gate, (cirq.CZPowGate,
                                         cirq.XPowGate,
                                         cirq.PhasedXPowGate,
                                         cirq.YPowGate)):
          raise ValueError('{!r} is not a supported gate'.format(operation.gate))
      if len(operation.qubits) == 2:
          p, q = operation.qubits
          if not p.is_adjacent(q):
            raise ValueError('Non-local interaction: {}'.format(repr(operation)))


  def validate_circuit(self, circuit):
      for moment in circuit:
          for operation in moment.operations:
              self.validate_operation(operation)
```
This device, for example, knows that two qubit gates between
next-nearest-neighbors is not valid:
```python
device = Xmon10Device()
circuit = cirq.Circuit()
circuit.append([cirq.CZ(device.qubits[0], device.qubits[2])])
try: 
  device.validate_circuit(circuit)
except ValueError as e:
  print(e)
# prints something like
# ValueError: Non-local interaction: Operation(cirq.CZ, (GridQubit(0, 0), GridQubit(2, 0)))
```
