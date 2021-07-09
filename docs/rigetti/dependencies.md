# Dependencies

Note, all executors with the exception of
:py`cirq_rigetti.circuit_sweep_executors.without_quilc_compilation`{.interpreted-text
role="func"} compile Quil to native gates using the [Quil
compiler](https://pyquil-docs.rigetti.com/en/stable/compiler.html). This
requires that you are running Quilc in order to execute a Cirq circuit
on a pyQuil `QuantumComputer`.

Additionally, you will need to install the pyQuil
[QVM](https://pyquil-docs.rigetti.com/en/stable/migration.html) unless
you are running against QPU hardware.

In order to install those dependencies, please see [Downloading the QVM
and
Compiler](https://pyquil-docs.rigetti.com/en/stable/start.html#sdkinstall).
Alternatively, you can run the
[docker-compose.test.yaml](https://github.com/quantumlib/cirq/cirq-rigetti/docker-compose.test.yaml)
file in this package\'s repository.