# ============================================
#   QNN Error Mitigation v2.0 — UPGRADED
#   Author: Akhil Cheela
#   Changes: 5 qubits, deeper circuit, 
#            noisier backend, better plots
# ============================================

# ── Imports ──────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.circuit.library import RealAmplitudes
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

print("✓ All imports successful!")

# ── Connect to IBM Quantum ────────────────────
QiskitRuntimeService.save_account(
    channel="ibm_quantum_platform",
    token="0rroWelX4XphgnV4hc1k3-9MXRwzhcGEsEZwTc-irX8j",
    overwrite=True
)

service = QiskitRuntimeService(channel="ibm_quantum_platform")

# Get ALL backends and pick noisiest one with short queue
backends = service.backends(operational=True, simulator=False)
print(f"\n{'Backend':<20} {'Qubits':<10} {'Queue':<10}")
print("-" * 40)
for b in backends:
    try:
        print(f"{b.name:<20} {b.num_qubits:<10} {b.status().pending_jobs:<10}")
    except:
        pass

# Pick least busy — we'll force more noise via deeper circuit
backend = service.least_busy(
    operational=True,
    simulator=False,
    min_num_qubits=5
)

print(f"\n✓ Connected!")
print(f"✓ Backend selected : {backend.name}")
print(f"✓ Qubits available : {backend.num_qubits}")
print(f"✓ Jobs in queue    : {backend.status().pending_jobs}")

# ── Build BIGGER Circuit (5 qubits, 4 layers) ─
N_QUBITS = 5
N_REPS   = 4   # deeper = more noise!

ansatz = RealAmplitudes(
    num_qubits=N_QUBITS,
    reps=N_REPS,
    entanglement="full"   # full entanglement = maximum noise
)
params = np.random.uniform(-np.pi, np.pi, ansatz.num_parameters)
bound_circuit = ansatz.assign_parameters(params)
bound_circuit.measure_all()

print(f"\n✓ Upgraded circuit built!")
print(f"✓ Qubits      : {N_QUBITS}  (was 3)")
print(f"✓ Layers      : {N_REPS}   (was 2)")
print(f"✓ Parameters  : {ansatz.num_parameters}  (was 9)")
print(f"✓ Entangle    : full  (was linear)")

# ── Ideal Distribution (Noiseless Simulator) ─
ideal_sim        = AerSimulator()
ideal_transpiled = transpile(bound_circuit, ideal_sim)
ideal_result     = ideal_sim.run(ideal_transpiled, shots=4096).result()
ideal_counts     = ideal_result.get_counts()

def counts_to_probs(counts, n_qubits=N_QUBITS):
    total = sum(counts.values())
    probs = np.zeros(2**n_qubits)
    for state, count in counts.items():
        clean = state.replace(" ", "")[-n_qubits:]
        probs[int(clean, 2)] = count / total
    return torch.tensor(probs, dtype=torch.float32)

ideal_probs = counts_to_probs(ideal_counts)
print(f"\n✓ Ideal distribution computed ({2**N_QUBITS} states)")
print("Top 5 states:")
top5 = np.argsort(ideal_probs.numpy())[-5:][::-1]
for i in top5:
    print(f"  |{i:05b}⟩ : {ideal_probs[i]:.4f}")

# ── Run on Real IBM Hardware ──────────────────
transpiled = transpile(
    bound_circuit,
    backend=backend,
    optimization_level=3
)

print(f"\n✓ Circuit depth after transpile : {transpiled.depth()}")
print(f"✓ Submitting to                 : {backend.name}")
print("⏳ Running on real quantum hardware...")

sampler = Sampler(mode=backend)
job     = sampler.run([transpiled], shots=4096)

print(f"✓ Job ID: {job.job_id()}")
print("⏳ Waiting for results...")

result      = job.result()
counts      = result[0].data.meas.get_counts()
noisy_probs = counts_to_probs(counts)

# Fidelity before mitigation
def fidelity(p, q):
    return float(np.sum(np.sqrt(np.abs(p)) * np.sqrt(np.abs(q)))**2)

print(f"\n✓ Real hardware results received!")
print(f"✓ Fidelity before mitigation : {fidelity(noisy_probs.numpy(), ideal_probs.numpy())*100:.2f}%")

# ── Train QNN Mitigator (bigger network) ─────
class QNNMitigator(nn.Module):
    def __init__(self, n_states):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128),      nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64),       nn.ReLU(),
            nn.Linear(64, n_states),  nn.Softmax(dim=-1)
        )
    def forward(self, x):
        return self.net(x)

N_STATES  = 2**N_QUBITS
mitigator = QNNMitigator(N_STATES)
optimizer = optim.Adam(mitigator.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
losses    = []

print("\n🧠 Training upgraded QNN Mitigator (100 epochs)...\n")

for epoch in range(100):
    pred = mitigator(noisy_probs.unsqueeze(0)).squeeze()
    loss = F.kl_div(pred.log(), ideal_probs, reduction='sum')
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()
    losses.append(loss.item())
    if epoch % 20 == 0:
        print(f"Epoch {epoch:03d} | Loss: {loss.item():.6f} | LR: {scheduler.get_last_lr()[0]:.6f}")

print("\n✓ Training complete!")

# ── Final Results ─────────────────────────────
mitigated_probs = mitigator(noisy_probs.unsqueeze(0)).squeeze().detach().numpy()

f_noisy     = fidelity(noisy_probs.numpy(),  ideal_probs.numpy())
f_mitigated = fidelity(mitigated_probs,      ideal_probs.numpy())
improvement = ((f_mitigated - f_noisy) / f_noisy) * 100

print("\n" + "=" * 50)
print(f"   FINAL RESULTS v2.0 — {backend.name}")
print("=" * 50)
print(f"  Qubits             : {N_QUBITS} (upgraded from 3)")
print(f"  Circuit depth      : {transpiled.depth()}")
print(f"  Shots              : 4096 (upgraded from 1024)")
print(f"  Noisy fidelity     : {f_noisy:.4f}  ({f_noisy*100:.2f}%)")
print(f"  Mitigated fidelity : {f_mitigated:.4f}  ({f_mitigated*100:.2f}%)")
print(f"  Improvement        : +{improvement:.2f}%")
print("=" * 50)

# ── Plot ──────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.patch.set_facecolor("#020b18")
fig.suptitle(f"QNN Error Mitigation v2.0 — {backend.name} | {N_QUBITS} Qubits | Depth {transpiled.depth()}",
             color="#00e5ff", fontsize=12, fontweight="bold")

# Top 16 states for readability
top16   = np.argsort(ideal_probs.numpy())[-16:][::-1]
states  = [f"|{i:05b}⟩" for i in top16]
x       = np.arange(len(top16))
w       = 0.28

ax1 = axes[0]
ax1.set_facecolor("#071428")
ax1.bar(x - w, ideal_probs.numpy()[top16],     w, label="Ideal",     color="#00e5ff", alpha=0.9)
ax1.bar(x,     noisy_probs.numpy()[top16],     w, label="Noisy QPU", color="#ef4444", alpha=0.8)
ax1.bar(x + w, mitigated_probs[top16],         w, label="Mitigated", color="#10b981", alpha=0.9)
ax1.set_xticks(x)
ax1.set_xticklabels(states, color="#4a7fa5", fontsize=7, rotation=45)
ax1.set_title("Top 16 State Distributions", color="#00e5ff", pad=10)
ax1.set_ylabel("Probability", color="#cce7ff")
ax1.tick_params(colors="#4a7fa5")
ax1.legend(facecolor="#071428", labelcolor="#cce7ff", fontsize=9)
for spine in ax1.spines.values():
    spine.set_edgecolor("#0d2d50")

# Loss curve
ax2 = axes[1]
ax2.set_facecolor("#071428")
ax2.plot(losses, color="#7c3aed", linewidth=2.5, label="QNN Loss")
ax2.fill_between(range(len(losses)), losses, alpha=0.15, color="#7c3aed")
ax2.axhline(y=losses[-1], color="#10b981", linestyle="--", linewidth=1.5,
            label=f"Final: {losses[-1]:.4f}")
ax2.set_title("Training Loss (100 epochs)", color="#00e5ff", pad=10)
ax2.set_xlabel("Epoch", color="#cce7ff")
ax2.set_ylabel("KL Divergence", color="#cce7ff")
ax2.tick_params(colors="#4a7fa5")
ax2.legend(facecolor="#071428", labelcolor="#cce7ff", fontsize=9)
for spine in ax2.spines.values():
    spine.set_edgecolor("#0d2d50")

# Fidelity comparison bar
ax3 = axes[2]
ax3.set_facecolor("#071428")
bars = ax3.bar(
    ["Noisy\n(Real QPU)", "Mitigated\n(QNN)", "Ideal\n(Target)"],
    [f_noisy * 100, f_mitigated * 100, 100],
    color=["#ef4444", "#10b981", "#00e5ff"],
    alpha=0.85, width=0.5
)
for bar, val in zip(bars, [f_noisy*100, f_mitigated*100, 100]):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f"{val:.2f}%", ha="center", color="#cce7ff", fontsize=10, fontweight="bold")
ax3.set_title(f"Fidelity Comparison\n+{improvement:.2f}% improvement",
              color="#00e5ff", pad=10)
ax3.set_ylabel("Fidelity (%)", color="#cce7ff")
ax3.set_ylim(0, 110)
ax3.tick_params(colors="#4a7fa5")
for spine in ax3.spines.values():
    spine.set_edgecolor("#0d2d50")

plt.tight_layout()
plt.savefig("qnn_results_v2.png", dpi=150, bbox_inches="tight")
plt.show()
print("✓ Plot saved as qnn_results_v2.png")