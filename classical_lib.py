import torch
from torch import nn
import cirq
import numpy as np

#torch.manual_seed(0)

# Layer of single qubit z rotations
def rot_z_layer(n_qubits, parameters):
    if n_qubits != len(parameters):
        raise ValueError("Too many or few parameters, must equal n_qubits")
    for i in range(n_qubits):
        yield cirq.rz(2 * parameters[i])(cirq.GridQubit(i, 0))

# Layer of single qubit y rotations
def rot_y_layer(n_qubits, parameters):
    if n_qubits != len(parameters):
        raise ValueError("Too many of few parameters, must equal n_qubits")
    for i in range(n_qubits):
        yield cirq.ry(parameters[i])(cirq.GridQubit(i, 0))

# Layer of entangling CZ(i,i+1 % n_qubits) gates.
def entangling_layer(n_qubits):
    if n_qubits == 1:
        return
    if n_qubits == 2:
        yield cirq.CZ(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0))
        return
    for i in range(n_qubits):
        yield cirq.CZ(cirq.GridQubit(i, 0), cirq.GridQubit((i+1) % n_qubits, 0))

# Variational circuit, i.e., the ansatz.
def variational_circuit(n_qubits, depth, theta):
    if len(theta) != (2 * depth * n_qubits):
        raise ValueError("Theta of incorrect dimension, must equal 2*depth*n_qubits")

    # Initializing qubits and circuit
    qubits = [cirq.GridQubit(i, 0) for i in range(n_qubits)]
    circuit = cirq.Circuit()

    # Adding layers of rotation gates and entangling gates.
    for d in range(depth):
        # Adding single qubit rotations
        circuit.append(rot_z_layer(n_qubits, theta[d * 2 * n_qubits : (d+1) * 2 * n_qubits : 2]))
        circuit.append(rot_y_layer(n_qubits, theta[d * 2 * n_qubits + 1 : (d+1) * 2 * n_qubits + 1 : 2]))
        # Adding entangling layer
        circuit.append(entangling_layer(n_qubits))

    # Adding measurement at the end.
    circuit.append(cirq.measure(*qubits, key='m'))
    return circuit

def estimate_probs(circuit, theta, n_qubits, n_shots=10000):
    # Creating parameter resolve dict by adding state and theta.
    try:
        theta_mapping = [('theta_' + str(i), theta[i]) for i in range(len(theta))]
    except IndexError as error:
        print("Could not resolve theta symbol, array of wrong size.")
    resolve_dict = dict(theta_mapping)
    resolver = cirq.ParamResolver(resolve_dict)
    resolved_circuit = cirq.resolve_parameters(circuit, resolver)

    # Run the circuit.
    results = cirq.sample(resolved_circuit, repetitions=n_shots)
    frequencies = results.histogram(key='m')
    probs = np.zeros(2**n_qubits)
    for key, value in frequencies.items():
        probs[key] = value / n_shots
    return probs

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64,1)
            nn.Sigmoid(),
        )

    def forward(self, x):
        output = self.model(x)
        return output

class Generator:
    def __init__(self, circuit, theta, n_qubits):
        self.circuit = circuit
        self.theta = theta
        self.n_qubits = n_qubits

    def estimate_probs(self):
        # Creating parameter resolve dict by adding state and theta.
        try:
            theta_mapping = [('theta_' + str(i), self.theta[i]) for i in range(len(self.theta))]
        except IndexError as error:
            print("Could not resolve theta symbol, array of wrong size.")
        resolve_dict = dict(theta_mapping)
        resolver = cirq.ParamResolver(resolve_dict)
        resolved_circuit = cirq.resolve_parameters(self.circuit, resolver)

        # Run the circuit.
        results = cirq.sample(resolved_circuit, repetitions=10**3)
        frequencies = results.histogram(key='m')
        probs = np.zeros((2**self.n_qubits, 2))
        for key, value in frequencies.items():
            probs[key, 0] = key
            probs[key, 1] = value / len(frequencies)
        return probs

    def forward(self, index):
        key = index
        prob = self.estimate_probs()[index, 1]
        return key, prob

def real_probabs(n_qubits):
    theta = np.random.random(2*n_qubits)*2*np.pi
    real_circ = variational_circuit(n_qubits, 1, theta)
    probs = estimate_probs(real_circ, theta, n_qubits)
    return probs
