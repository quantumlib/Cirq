'''
---------------------------------
SIMONS'S ALGORITHM - OVERVIEW
---------------------------------
Simon's Algorithm solves the problem of finding a particular value of s when some function f:{0, 1}^n --> {0, 1}^n
is inputted into the program that follows this rule: "f(x) = f(y) if and only if x (+) y is in the set {0^n, s}"
---------------------------------
STEPS OF THE ALGORITHM
---------------------------------
1. Begin with two n-qubit registers, each in the |0> state
2. Apply a Hadamard transform to the first n-qubit register, therefore creating an even superposition of states
3. The oracle (which encodes values of the function) is queried B_f |x>|y> = |x>|y (+) f(x)>, therefore mapping |0^n> --> |f(x)>
4. Apply a Hadamard transform to the first n-qubit register
---------------------------------
MEASUREMENT
---------------------------------
It can be found that for an output string y, then y (dot mod 2) s is always equal to 0, as we can calculate y (dot mod 2) s = 1 occuring with probability = 0
We get a system of eqautions, which we can use to solve for s (provided y_1, ..., y_(n-1) are linearlly independent)! We measure the string y to be the first n-qubit register
'''

import cirq
import random
import numpy as np
import copy
import sympy
import itertools

# Qubit preparation

number_qubits = #Number of qubits

def main(number_qubits):

    circuit_sampling = number_qubits-1

    #Create the qubits which are used within the circuit

    first_qubits = [cirq.GridQubit(i, 0) for i in range(number_qubits)]
    second_qubits = [cirq.GridQubit(i, 0) for i in range(number_qubits, 2*number_qubits)]

    the_activator = cirq.GridQubit(2*number_qubits, 0)

    #Create the qubits that can be used for large-input Toffoli gates
    ancilla = []
    for v in range(2*number_qubits+1, 3*number_qubits):
        ancilla.append(cirq.GridQubit(v, 0))

    #Create the function that is inputted into the algorithm (secret!)

    domain = []
    selector = []
    co_domain = []
    fixed = []

    for k in range(0, 2**number_qubits):
        domain.append(k)
        selector.append(k)
        co_domain.append(False)
        fixed.append(k)

    #Create the "secret string"
    s = domain[random.randint(0, len(domain)-1)]

    #Create the "secret function"
    for g in range(0, int((2**number_qubits)/2)):
        v = random.choice(selector)
        x = random.choice(domain)
        co_domain[x] = v
        co_domain[x^s] = v
        del selector[selector.index(v)]
        del domain[domain.index(x)]
        if (s != 0):
            del domain[domain.index(x^s)]

    secret_function = [fixed, co_domain]

    oracle = make_oracle(ancilla, secret_function, first_qubits, second_qubits, s, the_activator)

    c = make_simon_circuit(first_qubits, second_qubits, oracle)

    #Sampling the circuit

    simulator = cirq.Simulator()
    result = simulator.run(c, repetitions=number_qubits-1)
    final = result.histogram(key='y')
    print("Secret String: "+str(s))
    print("Secret Function (Domain and Co-Domain): "+str(secret_function))
    final = str(result)[str(result).index("y")+2:len(str(result))].split(", ")
    last = []
    for i in range(0, number_qubits-1):
        holder = []
        for j in final:
            holder.append(int(j[i]))
        holder.append(0)
        last.append(holder)

    print("Results: "+str(last))
    return [last, secret_function, s, last]




def make_oracle(ancilla, secret_function, first_qubits, second_qubits, s, the_activator):

    #Hard-code oracle on a case-by-case basis


    for o in range(0, len(secret_function[0])):
        counter = 0
        for j in list(str(format(secret_function[0][o], "0"+str(number_qubits)+"b"))):
            if (int(j) == 0):
                yield cirq.X.on(first_qubits[counter])
            counter = counter+1
        yield apply_n_qubit_tof(ancilla, first_qubits+[the_activator])
        counter = 0
        for j in list(str(format(secret_function[0][o], "0"+str(number_qubits)+"b"))):
            if (int(j) == 0):
                yield cirq.X.on(first_qubits[counter])
            counter = counter+1

        counter = 0
        for j in list(str(format(secret_function[1][o], "0"+str(number_qubits)+"b"))):
            if (int(j) == 1):
                yield cirq.CNOT.on(the_activator, second_qubits[counter])
            counter = counter+1

        counter = 0
        for j in list(str(format(secret_function[0][o], "0"+str(number_qubits)+"b"))):
            if (int(j) == 0):
                yield cirq.X.on(first_qubits[counter])
            counter = counter+1
        yield apply_n_qubit_tof(ancilla, first_qubits+[the_activator])
        counter = 0
        for j in list(str(format(secret_function[0][o], "0"+str(number_qubits)+"b"))):
            if (int(j) == 0):
                yield cirq.X.on(first_qubits[counter])
            counter = counter+1

def apply_n_qubit_tof(ancilla, args):

    if (len(args) == 3):
        yield cirq.CCX.on(args[0], args[1], args[2])

    else:

        yield cirq.CCX.on(args[0], args[1], ancilla[0])
        for k in range(2, len(args)-1):
            yield cirq.CCX(args[k], ancilla[k-2], ancilla[k-1])

        yield cirq.CNOT.on(ancilla[len(args)-3], args[len(args)-1])

        for k in range(len(args)-2, 1, -1):
            yield cirq.CCX(args[k], ancilla[k-2], ancilla[k-1])
        yield cirq.CCX.on(args[0], args[1], ancilla[0])


def make_simon_circuit(first_qubits, second_qubits, oracle):

    circuit = cirq.Circuit()

    #Apply the first set of Hadamard gates

    for i in range(0, number_qubits):
        circuit.append(cirq.H.on(first_qubits[i]))

    #Apply the oracle

    circuit.append(oracle)

    #Apply the second set of Hadamard gates

    for i in range(0, number_qubits):
        circuit.append(cirq.H.on(first_qubits[i]))

    #Perform measurements upon the qubits

    circuit.append(cirq.measure(*second_qubits, key='x'))
    circuit.append(cirq.measure(*first_qubits, key='y'))

    return circuit

run = main(number_qubits)
matrix_input = run[0]
secret_function = run[1]
string_secret = run[2]
r = run[3]

def shuffle_op(matrix, point):


    for i in range(0, len(matrix)):
        if (matrix[i] == [0 for l in range(0, len(matrix[0]))]):
            raise ValueError("System of equations not linearly independent, try again")
        for j in range(0, len(matrix)):
            if (matrix[i] == matrix[j] and i != j):
                raise ValueError("System of equations not linearly independent, try again")


    for i in range(1, len(matrix)+1):
      for c in list(itertools.combinations([y[0:len(matrix)+1] for y in matrix], i)):
        hol = []
        for b in c:
          hol.append(b)
        calc = [sum(x)%2 for x in zip(*hol)]
        ha = True
        for ij in range (0, len(hol)):
            for ik in range (0, len(hol)):
                if (hol[ik] == hol[ij] and ik != ij):
                    ha = False
        if (ha == True and calc == [0 for p in range(0, len(calc))]):
          raise ValueError("System of equations not linearly independent, try again")

    flip = False
    passage = False

    for i in range(0, len(matrix)):
        for j in range(i+1, len(matrix)):
            if (matrix[i][i] != 0):
                x = -1*matrix[j][i]/matrix[i][i]
                iterator = map(lambda y: y*int(x), matrix[i])
                new = [sum(z)%2 for z in zip(matrix[j], iterator)]
                matrix[j] = new


    for a in range(0, len(matrix)+1):

        fml = []
        flip = a

        work = copy.deepcopy(matrix)

        h = [0 for i in range(0, len(matrix[0])-2)]+[point]
        h.insert(flip, 1)
        work.append(h)

        for j in range(0, len(work[0])-1):
            temporary = []
            for g in range(0, len(work)):
                if (work[g][j] == 1):
                    temporary.append(work[g])
            fml.append(temporary)

        cv = False

        if ([] not in fml):
            for element in itertools.product(*fml):

                if (sorted(work) == sorted(element)):

                    cv = True

                    last_work = copy.deepcopy(list(element))

                    #Check for linear independence

                    for i in range(0, len(last_work)):
                        if (last_work[i] == [0 for l in range(0, len(last_work[0]))]):
                            cv = False
                        for j in range(0, len(last_work)):
                            if (last_work[i] == last_work[j] and i != j):
                                cv = False


                    for i in range(1, len(last_work)+1):
                      for c in list(itertools.combinations([y[0:len(last_work)] for y in last_work], i)):
                        hol = []
                        for b in c:
                          hol.append(b)
                        calc = [sum(x)%2 for x in zip(*hol)]
                        ha = True
                        for ij in range (0, len(hol)):
                            for ik in range (0, len(hol)):
                                if (hol[ik] == hol[ij] and ik != ij):
                                    ha = False
                        if (ha == True and calc == [0 for p in range(0, len(calc))]):
                          cv = False

                    #Check if the matrix can be reduced

                    for i in range(0, len(last_work)):
                        for j in range(i+1, len(last_work)):
                            if (last_work[i][i] == 0):
                                cv = False
                            else:
                                x = -1*last_work[j][i]/last_work[i][i]
                                iterator = map(lambda y: y*int(x), last_work[i])
                                new = [sum(z)%2 for z in zip(last_work[j], iterator)]
                                last_work[j] = new

                    if (cv == True):
                        break;


        if (cv == True):
            break;

    matrix = last_work

    return last_work

def construct_solve(matrix_out):

    final_matrix = matrix_out

    solution = []

    for i in range(len(final_matrix)-1, 0, -1):
      solution.append(final_matrix[i][len(final_matrix[i])-1])
      for j in range(0, i):
        if (final_matrix[j][i] == 1):
          final_matrix[j][len(final_matrix[i])-1] = (final_matrix[j][len(final_matrix[i])-1]-final_matrix[i][len(final_matrix[i])-1])%2
    solution.append(final_matrix[0][len(final_matrix[i])-1])

    solution.reverse()

    return solution


other_matrix_input = copy.deepcopy(matrix_input)

first_shot = shuffle_op(matrix_input, 0)
try_1 = construct_solve(first_shot)
second_shot = shuffle_op(other_matrix_input, 1)
try_2 = construct_solve(second_shot)

processing1 = ''.join(str(x) for x in try_1)
processing2 = ''.join(str(x) for x in try_2)

the_last = 0
if (secret_function[1][secret_function[0].index(0)] == secret_function[1][secret_function[0].index(int(processing1, 2))] and secret_function[1][secret_function[0].index(0)] == secret_function[1][secret_function[0].index(int(processing2, 2))]):
    final = int(processing1, 2)
    if (int(processing1, 2) == 0):
        final = int(processing2, 2)
    print("The secret string is: "+str(final))
    the_last = final

else:
    print("The secret string is 0")
    the_last = 0
