from qgan_lib import *
import cirq
import numpy as np

epochs = 50

n_qubits = 1
depth = 1
n_params = 2*depth*n_qubits

theta0 = np.random.random(size=[n_qubits, 2])*2*np.pi

#real_state = cirq.testing.random_superposition(dim=2**n_qubits)
real_state = cirq.final_state_vector(real_circuit(n_qubits, theta0))

gen = Generator(n_qubits, depth, theta0)

dis = Discriminator(n_qubits)

f = compute_fidelity(gen, real_state)

while(f < 0.99):
    for epoch in range(epcohs):
        
