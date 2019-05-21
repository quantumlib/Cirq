from cirq.aqt import AQTSampler, AQTSamplerSim
from cirq.aqt.aqt_device import get_aqt_device
from cirq import LineQubit,X,XX,Circuit, measure, study
import sympy
import numpy as np

# def test_aqt_sampler():
#      #TODO: create AQT ion trap object with given number of qubits
#      theta = sympy.Symbol('theta')
#      num_points =10
#      max_angle = np.pi
#      repetitions = 10
#      sampler = AQTSampler()
#      qubit = LineQubit(0)
#      circuit = Circuit.from_ops(X(qubit) ** theta)
#      circuit.append(measure(qubit, key='z'))
#      sweep = study.Linspace(key='theta', start=0.1, stop=max_angle / np.pi,
#                             length=num_points)
#      results = sampler.run_sweep(circuit, params=sweep, repetitions=repetitions, no_qubit=1)
#      angles = np.linspace(0.0, max_angle, num_points)
#      excited_state_probs = np.zeros(num_points)
#      for i in range(num_points):
#          excited_state_probs[i] = np.mean(results[i].measurements['m'])
#      print(excited_state_probs)

def test_aqt_sampler_sim():
    theta = sympy.Symbol('theta')
    num_points =10
    max_angle = np.pi
    repetitions = 100
    no_qubit = 4
    device, qubits = get_aqt_device(no_qubit)
    sampler = AQTSamplerSim()
    sampler.simulate_ideal = True
    circuit = Circuit.from_ops(X(qubits[3]) ** theta)
    sweep = study.Linspace(key='theta', start=0.1, stop=max_angle / np.pi,
                           length=num_points)
    results = sampler.run_sweep(circuit, params=sweep, repetitions=repetitions, no_qubit=no_qubit)
    angles = np.linspace(0.0, max_angle, num_points)
    excited_state_probs = np.zeros(num_points)
    #print(results)
    for i in range(num_points):
        excited_state_probs[i] = np.mean(results[i].measurements['m'])
    print(excited_state_probs[-1])
    assert excited_state_probs[-1] == 0.25


def test_aqt_sampler_sim_xtalk():
    theta = sympy.Symbol('theta')
    num_points =10
    max_angle = np.pi
    repetitions = 100
    no_qubit = 4
    device, qubits = get_aqt_device(no_qubit)
    sampler = AQTSamplerSim()
    sampler.simulate_ideal = False
    circuit = Circuit.from_ops(X(qubits[0]),X(qubits[3]),X(qubits[2]))
    sweep = study.Linspace(key='theta', start=0.1, stop=max_angle / np.pi,
                           length=num_points)
    results = sampler.run_sweep(circuit, params=sweep, repetitions=repetitions, no_qubit=no_qubit)
    angles = np.linspace(0.0, max_angle, num_points)
    excited_state_probs = np.zeros(num_points)


def test_aqt_sampler_ms():
    #TODO: Check big/little endian of result
    repetitions = 100
    no_qubit = 4
    device, qubits = get_aqt_device(no_qubit)
    sampler = AQTSamplerSim()
    circuit = Circuit(device=device)
    for i in range(9):
        circuit.append(XX(qubits[0],qubits[1]) ** 0.5)
    results = sampler.run(circuit, repetitions=repetitions, no_qubit=no_qubit)
    hist = (results.histogram(key='m'))
    print(hist)
    assert hist[12] > repetitions/3
    assert hist[0] > repetitions/3
