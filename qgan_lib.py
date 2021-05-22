import cirq
import numpy as np
from itertools import product

# Layer of single qubit z rotations
def rot_z_layer(n_qubits, parameters):
    if n_qubits != len(parameters):
        raise ValueError("Too many or few parameters, must equal n_qubits")
    for i in range(n_qubits):
        yield cirq.rz(2 * parameters[i])(cirq.GridQubit(i, 0))

# Layer of single qubit y rotations
def rot_y_layer(n_qubits, parameters):
    if n_qubits != len(parameters):
        raise ValueError("Too many or few parameters, must equal n_qubits")
    for i in range(n_qubits):
        yield cirq.ry(parameters[i])(cirq.GridQubit(i, 0))

# Layer of entangling CZ(i,i+1 % n_qubits) gates.
def entangling_layer(n_qubits):
    if n_qubits == 1:
        pass
    elif n_qubits == 2:
        yield cirq.CZ(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0))
        return
    for i in range(n_qubits):
        yield cirq.CZ(cirq.GridQubit(i, 0), cirq.GridQubit((i+1) % n_qubits, 0))

# Variational circuit, i.e., the ansatz.
def qcbm(n_qubits, depth, theta):
    if len(theta) != (2 * depth * n_qubits):
        raise ValueError("Theta of incorrect dimension, must equal 2*depth*n_qubits")

    # Initializing qubits and circuit
    qubits = [cirq.GridQubit(i, 0) for i in range(n_qubits)]
    circuit = cirq.Circuit()

    # Adding layers of rotation gates and entangling gates.
    for d in range(depth):
        # Adding single qubit rotations
        circuit.append(rot_z_layer(n_qubits, theta[:,d,0]))
        circuit.append(rot_y_layer(n_qubits, theta[:,d,1]))
        # Adding entangling layer
        circuit.append(entangling_layer(n_qubits))

    # Adding measurement at the end.
    #circuit.append(cirq.measure(*qubits, key='m'))
    return circuit

def compute_fidelity(gen, real_state):
    fake_state = gen.produce_fake_state()
    fidelity = cirq.qis.fidelity(fake_state, real_state)
    return fidelity

class Generator:
    def __init__(self, n_qubits, depth, theta):
        self.n_qubits = n_qubits
        self.depth = depth
        self.params = theta
        self.qc = qcbm(n_qubits, depth, self.params)

    def produce_fake_state(self):
        fake_state = cirq.final_state_vector(self.qc)
        return fake_state


class Discriminator:
    def __init__(self, n_qubits):
        self.alpha = -1 + 2*np.random.random(4**n_qubits)
        self.beta = -1 + 2*np.random.random(4**n_qubits)


    def grad_alpha(self, gen, real_state):
        n_qubits = gen.n_qubits
        qubits = cirq.LineQubit.range(n_qubits)
        grad_alpha_list = []
        circuit = gen.qc
        for pauli_index in range(4):
            for i in range(n_qubits):
                for j in range(n_qubits):
                    


def real_circuit(n_qubits, theta):
    qubits = cirq.LineQubit.range(n_qubits)
    circuit = cirq.Circuit()

    circuit.append(rot_z_layer(n_qubits, theta[:,0]))
    circuit.append(rot_y_layer(n_qubits, theta[:,1]))
    # Adding entangling layer
    circuit.append(entangling_layer(n_qubits))

    return circuit

def pauli_on_qubit(pauli_index, qubit_index):
    if pauli_index == 1:
        yield cirq.X(cirq.LineQubit(qubit_index))
        return
    if pauli_index == 2:
        yield cirq.Y(cirq.LineQubit(qubit_index))
        return
    if pauli_index == 3:
        yield cirq.Z(cirq.LineQubit(qubit_index))
        return
    else:
        raise ValueError("Pauli index out of range")

def compute_cost(gen, dis, theta0):
    fake_state = gen.produce_fake_state()
    n_qubits = gen.n_qubits

    alphas = dis.alpha
    cost = 0
    circuit = real_circuit(n_qubits, theta0)
    real_state = cirq.final_state_vector(circuit)
    for index, alpha in enumerate(alphas):
        circuit = real_circuit(n_qubits, theta0)
        for j in range(n_qubits):
            pauli_index = (index//(4**j))%4
            if pauli_index:
                circuit.append(pauli_on_qubit(pauli_index, j))
        state = cirq.final_state_vector(circuit)
        cost += alpha*np.asscalar( np.matmul( real_state.getH(), state ) )

    betas = dis.beta
    depth = gen.depth
    theta_gen = gen.params
    circuit = qcbm(n_qubits, depth, theta_gen)
    fake_state = gen.produce_fake_state()
    for index, beta in enumerate(betas):
        circuit = qcbm(n_qubits, depth, theta_gen)
        for j in range(n_qubits):
            pauli_index = (index//(4**j))%4
            if pauli_index:
                circuit.append(pauli_on_qubit(pauli_index, j))
        state = cirq.final_state_vector(circuit)
        cost += beta*np.asscalar( np.matmul( real_state.getH(), state ) )
    return cost
