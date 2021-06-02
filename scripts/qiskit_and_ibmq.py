import qiskit as q
from qiskit import IBMQ, QuantumCircuit, QuantumRegister
from qiskit.tools.monitor import job_monitor
from qiskit.visualization import plot_histogram
from qiskit.quantum_info.operators import Operator
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit import Aer
import matplotlib.pyplot as plt
import numpy as np

'''
Variables
'''
qubits_n = 2
accuracy = 10

'''
Circuits
'''
# Boring Circuit
boring_circuit = q.QuantumCircuit(qubits_n, qubits_n)    # 2 qubits, 2 bits
boring_circuit.x(0)            # NOT gate
boring_circuit.cx(0, 1)        # CNOT gate
boring_circuit.measure([0, 1], [0, 1])     # ([qubit register], [classical bit register])
boring_circuit.draw(output="mpl")


# Less Boring Circuit
less_boring_circuit = q.QuantumCircuit(qubits_n, qubits_n)
less_boring_circuit.h(0)             # Hadamard gate
less_boring_circuit.cx(0, 1)
less_boring_circuit.measure([0, 1], [0, 1])
less_boring_circuit.draw(output="mpl")


# Neutrino 2 Flavour Oscillation Circuit
class NeutrinoCircuit:
    def __init__(self):
        theta, phi = Parameter('theta'), Parameter('phi')

        theta_values = np.arange(start=0, stop=2 * np.pi, step=np.pi/accuracy).tolist()
        phi_values = np.arange(start=0, stop=np.pi, step=np.pi/accuracy).tolist()

        controls = QuantumRegister(4)
        circuit3 = QuantumCircuit(controls)
        circuit3.x(1)
        circuit3.x(3)
        circuit3.ry(theta=-theta, qubit=2, label='PMNS_H')
        circuit3.ry(theta=-theta, qubit=3, label='PMNS_H')

        # circuit3.unitary(PMNS_H, [2], label='PMNS_H')
        # circuit3.unitary(PMNS_H, [3], label='PMNS_H')

        # circuit3.unitary(U_t, [0], label='U(t)')
        # circuit3.unitary(U_t, [1], label='U(t)')
        # circuit3.unitary(U_t, [2], label='U(t)')
        # circuit3.unitary(U_t, [3], label='U(t)')

        bound_circuit = circuit3.bind_parameters({theta: theta_values})

        circuit3.measure([0, 1, 2, 3], [0, 1, 2, 3])
        circuit3.draw(output="mpl")

    def operators(self):
        theta = np.pi / 2
        phi = np.pi
        PMNS = Operator([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        PMNS_H = Operator([
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)]
        ])
        U_t = Operator([
            [1, 0],
            [0, np.exp(np.imag * phi)]
        ])


# Choose which circuit to use
circuit = less_boring_circuit

'''
IBM-Q Settings and Backend Access
'''
# Loads account
IBMQ.save_account(open("token.txt", "r").read())
IBMQ.load_account()

# Gets IBMQ providers
IBMQ.providers()
provider = IBMQ.get_provider("ibm-q")

# Tells us the queue length and qubit availability of all the backends
for backend in provider.backends():
    try:
        qubit_count = len(backend.properties().qubits)
    except:
        qubit_count = "simulated"
    print(f"{backend.name()} has {backend.status().pending_jobs} queued and {qubit_count} qubits")

# Simulator backend
for backend in Aer.backends():
    print(backend)


# Choose a specific backend
simulator_backend = Aer.get_backend('qasm_simulator')

backend = provider.get_backend("ibmq_athens")
backend_online_sim = provider.get_backend("ibmq_qasm_simulator")

'''
Run Simulation Through Backend
'''
job = q.execute(circuit, backend=backend_online_sim, shots=500)
job_monitor(job)

# Plot Results
result = job.result()
counts = result.get_counts(circuit)

plot_histogram([counts], legend=['Device'])

plt.show()











