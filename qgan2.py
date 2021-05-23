from qgan_lib2 import *
import numpy as np
import matplotlib.pyplot as plt

print(f'Setting up hyper parameters')
epochs = 30000
n_qubits = 2
gen_dis_ratio = 1

def construct_qcbm(circuit, n_qubits, depth):
    """
    Adds parametrised rotations about X and Z and entangling layer to a circuit.

    Parameters
    ----------
    circuit : Circuit
        An instance of circuit class to which gates are added.
    n_qubits : int
        Number of qubits in the system.
    depth : int
        Number of layers of gates. 1 layer consists of Rz and R_x on each qubit followed by CNOT on each neighbour pair.

    Returns
    -------
    circuit : Circuit
        The modified circuit
    """

    for d in range(depth):
        for i in range(n_qubits):
            circuit.append_gate(Gate('X', target = i, angle = np.random.random()*np.pi*2))
            circuit.append_gate(Gate('Z', target = i, angle = np.random.random()*np.pi*2))
        if n_qubits != 1:
            for i in range(n_qubits):
                circuit.append_gate(Gate('CNOT', control = i, target = (i+1)%n_qubits))
        for gate in circuit.gates:
            print(gate.angle)
    return circuit

init_state = initial_state(n_qubits)

real_circuit = Circuit(n_qubits)
real_circuit = construct_qcbm(real_circuit, n_qubits, 1)
for gate in real_circuit.gates:
    gate.angle = np.pi/2
real_state = np.matmul(real_circuit.circ_matrix(),init_state)

gen = Generator(n_qubits)
gen.circ = construct_qcbm(gen.circ, n_qubits, 2)

dis = Discriminator(n_qubits)
dis.randomize_disc()

fid = fidelity(gen, real_state)

loss_list = []
fid_list = []

while fid > 0.9 or fid < 0.15:
    gen.circ.randomize_angles()
    fid = fidelity(gen, real_state)
    print(f'Resetting angles because bad intialisation')

print('Starting training')
for iteration in range(epochs):
    #print(f'Iteration number {iteration + 1}')
    if iteration % gen_dis_ratio == 0:
        gen.update_params(dis)
    dis.update_params(gen, real_state)

    cost = np.around(compute_cost(gen, dis, real_state), 5)
    fid = np.around(fidelity(gen, real_state), 5)

    loss_list.append(cost)
    fid_list.append(fid)

    if (iteration+1) % 100 == 0:
        print(f'Iteration number {iteration + 1}, cost is {cost}, fidelity is {fid}')

    if fid > 0.99:
        break

    if iteration/epochs > 0.1 and fid_list[-1] == fid_list[-100]:
        break

plt.plot(loss_list)
plt.title('Loss vs Iterations')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()
plt.plot(fid_list)
plt.title('Fidelity vs Iterations')
plt.xlabel('Iteration')
plt.ylabel('Fidelity')
plt.ylim([-0.2, 1.2])
plt.show()
