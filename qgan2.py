from qgan_lib2 import *
import numpy as np

print(f'Setting up hyper parameters')
epochs = 300
n_qubits = 1
lr = 0.1

def construct_qcbm(circuit, n_qubits, depth):
    for d in range(depth):
        for i in range(n_qubits):
            circuit.append_gate(Gate('X', target = i, angle = np.random.random()*np.pi*2))
            circuit.append_gate(Gate('Z', target = i, angle = np.random.random()*np.pi*2))
        if n_qubits != 1:
            for i in range(n_qubits):
                circuit.append_gate(Gate('CNOT', control = i, target = (i+1)%n_qubits))
    print(circuit)
    return circuit

init_state = initial_state(n_qubits)

real_circuit = Circuit(n_qubits)
real_circuit = construct_qcbm(real_circuit, n_qubits, 1)
real_state = np.matmul(real_circuit.circ_matrix(),init_state)
print(f'Real state is {real_state}')

gen = Generator(n_qubits)
gen.circ = construct_qcbm(gen.circ, n_qubits, 2)

dis = Discriminator(n_qubits)

fid = fidelity(gen, real_state)

for iteration in range(epochs):
    print(f'Iteration number {iteration + 1}')
    gen.update_params(dis)
    dis.update_params(gen, real_state)

    cost = compute_cost(gen, dis, real_state)
    fid = fidelity(gen, real_state)
    if fid > 0.99:
        break
