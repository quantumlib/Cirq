import numpy as np
import pytest
import cirq
import classical_simulator 

class TestSimulator():

  def test_x_gate():
    q0 = cirq.LineQubit.range(1)
    circuit  = cirq.Circuit()
    circuit =  cirq.Circuit(cirq.X(q0), cirq.measure(cirq.LineQubit(q0), key='key'))
    expected_results = {'key': np.array([[[0]]], dtype=np.uint8)}
    sim = cirq.ClassicalSimulator()
    results = sim.run(circuit = circuit, param_resolver=None, repetitions = 1 )
    assert results == expected_results
    
  def test_CNOT():
    q0, q1 = cirq.LineQubit.range(2)
    circuit  = cirq.Circuit()
    circuit.append(cirq.X(q0))
    circuit.append(cirq.CNOT(q0, q1))
    circuit.append(cirq.measure(q1, key='key'))
    expected_results = {'key': np.array([[[1]]], dtype=np.uint8)}
    results = run(circuit = circuit, param_resolver=None, repetitions = 1 )
    assert results == expected_results

def test_measurment_gate():
  q0, q1 = cirq.LineQubit.range(2)
  circuit  = cirq.Circuit()
  circuit =  cirq.Circuit(cirq.measure(cirq.LineQubit(q0, q1), key='key'))
  expected_results = {'key': np.array[[[0, 0]]]}
  results = run(circuit = circuit, param_resolver=None, repetitions = 1 )
  assert results == expected_results


def test_qubit_order():
  q0, q1 = cirq.LineQubit.range(2)
  circuit  = cirq.Circuit()
  circuit =  cirq.Circuit(cirq.X(q0), cirq.CNOT(q0, q1), cirq.X(q0), cirq.measure(cirq.LineQubit(q1, q0), key='key'))
  expected_results = {'key': np.array[[[1,0]]]}
  results = run(circuit = circuit, param_resolver=None, repetitions = 1 )
  assert results == expected_results


def test_same_key_instances():
  q0, q1 = cirq.LineQubit.range(2)
  circuit  = cirq.Circuit()
  circuit =  cirq.Circuit(cirq.X(q0), cirq.measure(cirq.LineQubit(q0, q1), key='key'), cirq.X(q0), cirq.measure(cirq.LineQubit(q0, q1), key='key'))
  expected_results = {'key': np.array[[[1,0],[0,0]]]}
  results = run(circuit = circuit, param_resolver=None, repetitions = 1 )
  assert results == expected_results

def test_same_key_instances_order():
  q0, q1 = cirq.LineQubit.range(2)
  circuit  = cirq.Circuit()
  circuit =  cirq.Circuit(cirq.X(q0), cirq.measure(cirq.LineQubit(q0, q1), key='key'),  cirq.measure(cirq.LineQubit(q1, q0), key='key'))
  expected_results = {'key': np.array[[[1,0],[0,1]]]}
  results = run(circuit = circuit, param_resolver=None, repetitions = 1 )
  assert results == expected_results

def test_repetitions():
  q0, q1 = cirq.LineQubit.range(2)
  circuit  = cirq.Circuit()
  circuit =  cirq.Circuit(cirq.measure(cirq.LineQubit(q0), key='key'))
  expected_results = {'key': np.array[[[0]], [[0]], [[0]], [[0]], [[0]], [[0]], [[0]], [[0]], [[0]], [[0]]]}
  results = run(circuit = circuit, param_resolver=None, repetitions = 1 )
  assert results == expected_results

def test_multiple_gates():
  q0, q1 = cirq.LineQubit.range(2)
  circuit  = cirq.Circuit()
  circuit =  cirq.Circuit(cirq.X(q0), cirq.CNOT(q0, q1), cirq.CNOT(q0, q1), cirq.CNOT(q0, q1) cirq.measure(cirq.LineQubit(q0, q1), key='key'))
  expected_results = {'key': np.array[[[1,0]]]}
  results = run(circuit = circuit, param_resolver=None, repetitions = 1 )
  assert results == expected_results

def test_multiple_gates_order():
  q0, q1 = cirq.LineQubit.range(2)
  circuit  = cirq.Circuit()
  circuit =  cirq.Circuit(cirq.X(q0), cirq.CNOT(q0, q1), cirq.CNOT(q0, q1) cirq.measure(cirq.LineQubit(q0, q1), key='key'))
  expected_results = {'key': np.array[[[1,1]]]}
  results = run(circuit = circuit, param_resolver=None, repetitions = 1 )
  assert results == expected_results




        
    
