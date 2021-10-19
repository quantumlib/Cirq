import cirq

import cirq_superstaq


q0 = cirq.LineQubit(0)
q1 = cirq.LineQubit(1)

circuit = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1), cirq.measure(q0))


service = cirq_superstaq.Service(
    api_key="""Insert superstaq token that you received from https://superstaq.super.tech""",
    verbose=True,
)

job = service.create_job(circuit=circuit, repetitions=1, target="ibmq_qasm_simulator")
print("This is the job that's created ", job.status())

print(job.counts())
