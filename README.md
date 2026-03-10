# ⚛ Quantum Neural Network Error Mitigation

> Real quantum hardware error mitigation using hybrid classical-quantum neural networks

## 🏆 Results
| Version | Backend | Qubits | Noisy Fidelity | Mitigated Fidelity | Improvement |
|---------|---------|--------|---------------|-------------------|-------------|
| v1.0 | ibm_marrakesh | 3 | 98.5% | 99.1% | +0.6% |
| v2.0 | ibm_fez | 5 | 88.99% | 93.06% | +4.57% |
| v3.0 | Transformer QNN | ibm_fez | 5 | 56.90% | 99.80% | +75.40% |
## 🔬 Overview
A hybrid classical–quantum framework that learns to correct
readout errors, depolarizing noise and crosstalk from real
IBM Quantum hardware using variational quantum circuits
combined with deep neural mitigators.

## 🛠 Tech Stack
- IBM Quantum (ibm_fez, ibm_marrakesh)
- Qiskit + Qiskit Runtime
- PyTorch
- Python 3.12

## 📊 Architecture
- 5-qubit variational ansatz (4 layers, full entanglement)
- QNN: 4-layer neural network (256→128→64→32 units)
- Training: Adam optimizer, KL divergence loss, 100 epochs

## 🚀 Setup
```bash
pip install -r requirements.txt
python quantum_project.py
```

## 📁 Files
- `quantum_project.py` — main code
- `qnn_results_v2.png` — latest results plot
- `requirements.txt` — dependencies

## 👤 Author
**Akhil Cheela**