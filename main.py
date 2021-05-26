from qgan_lib import *
from components import *
import numpy as np
import matplotlib.pyplot as plt
import time

print(f'Setting up hyper parameters')
epochs = 1000
n_qubits = 1
gen_dis_ratio = 1
depth = 2

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
    return circuit

init_state = initial_state(n_qubits)

real_circuit = Circuit(n_qubits)
real_circuit = construct_qcbm(real_circuit, n_qubits, 1)
real_state = np.matmul(real_circuit.circ_matrix(),init_state)

gen = Generator(n_qubits)
gen.circ = construct_qcbm(gen.circ, n_qubits, depth)



dis = Discriminator(n_qubits)
dis.randomize_disc()

fid = fidelity(gen, real_state)

loss_list = []
gen_loss_list = []
fid_list = []

print('The fake circuit has starting parameters:')
for gate in gen.circ.gates:
    print(gate.angle)
print('Initiail fidelity is', fid)
print('Starting training')
tic = time.time()
for iteration in range(epochs):
    if (iteration+1) % gen_dis_ratio == 0:
        gen.update_params(dis)
    dis.update_params(gen, real_state)

    cost = np.around(compute_cost(gen, dis, real_state), 5)
    fid = np.around(fidelity(gen, real_state), 5)

    #gen_loss = np.around(gen_cost(gen, dis), 5)
    #gen_loss_list.append(gen_loss)

    loss_list.append(cost)
    fid_list.append(fid)

    if (iteration+1) % 100 == 0:
        print()
        print(f'Iteration number {iteration + 1}, cost is {cost}, fidelity is {fid}')
toc = time.time()
print(toc - tic)
#print('The fake circuit has final parameters:')
#for gate in gen.circ.gates:
#    print(gate.angle)

#print('The real circuit has parameters:')
#for gate in real_circuit.gates:
#    print(gate.angle)

plt.plot(loss_list)
plt.title('Discriminator Loss vs Iterations')
plt.xlabel('Iteration')
plt.ylabel('Discriminator Loss')
plt.show()
plt.plot(fid_list)
plt.title('Fidelity vs Iterations')
plt.xlabel('Iteration')
plt.ylabel('Fidelity')
plt.ylim([-0.2, 1.2])
plt.show()
#plt.plot(gen_loss_list)
#plt.title('Generator Loss vs Iterations')
#plt.xlabel('Iteration')
#plt.ylabel('Generator Loss')
#plt.show()
