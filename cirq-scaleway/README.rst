.. image:: https://www-uploads.scaleway.com/About_Generic_Hero_c4dc10a073.webp
  :target: https://github.com/quantumlib/cirq/
  :alt: cirq-scaleway
  :width: 500px

`Cirq <https://quantumai.google/cirq>`__ is a Python library for writing, manipulating, and optimizing quantum
circuits and running them against quantum computers and simulators.

This module is **cirq-scaleway**, which provides everything you'll need to run Cirq quantum algorithms on Scaleway Quantum as a Service (QaaS).

Documentation
-------------

To get started with Scaleway Quantum as a Service (QaaS), checkout the following guide and tutorial:

- `Access and authentication <https://quantumai.google/cirq/scaleway/access>`__
- `Getting started guide <https://quantumai.google/cirq/tutorials/scaleway/getting_started>`__

Installation
------------

To install the stable version of only **cirq-scaleway**, use `pip install cirq-scaleway`.

Note, that this will install both **cirq-scaleway** and **cirq-core**.

To get all the optional modules installed, you'll have to use `pip install cirq` or `pip install cirq~=1.0.dev` for the pre-release version.

Getting started
---------------

In the most simple way, here the code to use **cirq-scaleway**:

>>> import cirq
>>> from cirq_scaleway import ScalewayQuantumService

>>> service = ScalewayQuantumService(
    project_id="<your-scaleway-project-id>", secret_key="<your-scaleway-secret-key>"
)

>>> # Get and display all provided (real or simulated) devices
>>> devices = service.devices(min_num_qubits=34)
>>> print(devices)

>>> # Get a specific device by its name
>>> qsim_simulator = service.device(device="qsim_simulation_c64m512")

>>> # Create a device session and run a circuit against it
>>> with qsim_simulator.create_session():
>>>   qubit = cirq.GridQubit(0, 0)
>>>   circuit = cirq.Circuit(cirq.X(qubit) ** 0.5, cirq.measure(qubit, key='m'))

>>>   # Run the circuit on the device
>>>   result = sampler.run(circuit)
>>>   print(result)

Development
------------

Please see Cirq `development documentation <../docs/dev/development.md>`_.
