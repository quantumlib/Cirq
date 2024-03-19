def check_uniform_superposition_error(M, n):

    gate = generalized_uniform_superposition_gate(M, n)
    qregx = cirq.LineQubit.range(n)
    qcircuit = cirq.Circuit(gate.on(*qregx))
    
    unitary_matrix1 = np.real(qcircuit.unitary())

    np.testing.assert_allclose(
        unitary_matrix1[:,0],
        (1/np.sqrt(M))*np.array([1]*M + [0]*(2**n - M)),
        atol=1e-8,
    )

"""The following code tests the creation of M uniform superposition states, where M ranges from 3 to 1024."""
M=1025  
for mm in range(3, M):
    if (mm & (mm-1)) == 0:
        n = int(np.log2(mm))
    else:
        n = int(np.ceil(np.log2(M)))
    check_uniform_superposition_error(mm, n)
