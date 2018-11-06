## Schedules and Devices

``Schedule`` and ``Circuit`` are the two major container classes for
quantum circuits.  In contrast to ``Circuit``, a ``Schedule`` includes
detailed information about the timing and duration of the gates.

Conceptually a ``Schedule`` is made up of a set of ``ScheduledOperations``
as well as a description of the ``Device`` on which the schedule is
intended to be run.  Each ``ScheduledOperation`` is made up of a ``time``
when the operation starts and a ``duration`` describing how long the
operation takes, in addition to the ``Operation`` itself (like in a
``Circuit`` an ``Operation`` is made up of a ``Gate`` and the ``QubitIds``
upon which the gate acts.)

### Devices

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

  def duration_of(self, operation):
      # Wouldn't it be nice if everything took 10ns?
      return cirq.Duration(nanos=10)

  def validate_operation(self, operation):
      if not isinstance(operation, cirq.GateOperation):
          raise ValueError('{!r} is not a supported operation'.format(operation))
      if not isinstance(operation.gate, (cirq.CZPowGate,
                                         cirq.XPowGate,
                                         cirq.PhasedXPowGate,
                                         cirq.YPowGate,
                                         cirq.google.ExpWGate)):
          raise ValueError('{!r} is not a supported gate'.format(operation.gate))
      if len(operation.qubits) == 2:
          p, q = operation.qubits
          if not p.is_adjacent(q):
            raise ValueError('Non-local interaction: {}'.format(repr(operation)))


  def validate_scheduled_operation(self, schedule, scheduled_operation):
      self.validate_operation(scheduled_operation.operation)

  def validate_circuit(self, circuit):
      for moment in circuit:
          for operation in moment.operations:
              self.validate_operation(operation)

  def validate_schedule(self, schedule):
      for scheduled_operation in schedule.scheduled_operations:
          self.validate_scheduled_operation(schedule, scheduled_operation)
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

### Schedules

A ``Schedule`` contains more timing information above and beyond
that which is provided by the ``Moment`` structure of a ``Circuit``.
This can be used both for fine grained timing control, but also to
optimize a circuit for a particular device.  One can work directly
with ``Schedules`` or, more common, use a custom scheduler that
converts a ``Circuit`` to a ``Schedule``.  A simple example of
such a scheduler is the ``moment_by_moment_schedule`` method of
``schedulers.py``.  This scheduler attempts to keep the ``Moment``
structure of the underlying ``Circuit`` as much as possible:
each ``Operation`` in a ``Moment`` is scheduled to start at the 
same time (such a schedule may not be possible, in which case this
method raises an exception.)

Here, for example, is a simple ``Circuit`` on the ``Xmon10Device`` 
defined above
```python
from cirq.google.xmon_gates import ExpWGate
circuit = cirq.Circuit()
X = ExpWGate(exponent=1.0)
circuit.append([cirq.CZ(device.qubits[0], device.qubits[1]), X(device.qubits[0])])
print(circuit)
# prints:
# (0, 0): ───@───X───
#            │
# (1, 0): ───@───────
```
This can be converted over into a schedule using the moment by
moment schedule
```python
schedule = cirq.moment_by_moment_schedule(device, circuit)
```
Schedules have an attributed ``scheduled_operations`` which contains
all the scheduled operations in a ``SortedListWithKey``, where the
key is the start time of the ``SortedOperation``. ``Schedules`` support
nice helpers for querying about the time-space layout of the schedule.
For instance, the ``Schedule`` behaves as if it has an index corresponding
to time.  So, we can look up which operations occur at a specific time
```python
print(schedule[cirq.Timestamp(nanos=15)])
# prints something like 
# [ScheduledOperation(Timestamp(picos=10000), Duration(picos=10000),...)]
```
or even a start and end time using slicing notation
```python
slice = schedule[cirq.Timestamp(nanos=5):cirq.Timestamp(nanos=15)]
slice_schedule = cirq.Schedule(device, slice)
print(slice_schedule == schedule)
# prints True
```
More complicated queries across ``Schedules`` can be done using the
``query``.

``Schedules`` are usually built by converting from ``Circuits``,
but one can also directly manipulate the schedule using the 
``include`` and ``exclude`` methods.  ``include`` will check
if there are any collisions with other schedule operations.
