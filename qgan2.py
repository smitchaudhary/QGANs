from qgan_lib2 import *
import numpy as np

epochs = 50
n_qubits = 1
lr = 0.1

def construct_qcbm(circuit, n_qubits, depth):
    for d in range(depth):
        for i in range(n_qubits):
            circuit.append_gate(Gate('X', target = i, angle = np.random.random()*np.pi*2))
            circuit.append_gate(Gate('Y', target = i, angle = np.random.random()*np.pi*2))
        if n_qubits != 1:
            for i in range(n_qubits):
                circuit.append_gate(Gate('CNOT', control = i, target = (i+1)%n_qubits))

    return circuit

init_state = initial_state(n_qubits)

real_circuit = Circuit(n_qubits)
real_circuit = construct_qcbm(real_circuit, n_qubits, 1)
real_state = np.matmul(real_circuit.circ_matrix(),init_state)

gen = Generator(n_qubits)
gen.circ = construct_qcbm(gen.circ, n_qubits, 2)

dis = Discriminator(n_qubits)

fid = fidelity(gen, real_state)

while (fid < 0.99):
    for iteration in range(epochs):
        gen.update_params(dis)
        dis.update_params(gen, real_state)

        cost = compute_cost(gen, dis, real_state)
        fid = compute_fidelity(gen, real_state)
