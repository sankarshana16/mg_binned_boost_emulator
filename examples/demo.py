import numpy as np
import matplotlib.pyplot as plt

from emulator.linear_nn import LinearBoostNN
from emulator.gp_emulator import NonlinearBoostGP
from emulator.stitching import BoostEmulator


# -------------------------------------------------
# Paths (EDIT THESE if needed)
# -------------------------------------------------
MODEL_DIR = "/home/sankarshana/Codes/emulation_project_lmu/"

linear_model_path = MODEL_DIR + "linear_boost_nn.pt"

gp_full_path = MODEL_DIR + "gp_full_corrected.cpk"
pca_full_path = MODEL_DIR + "pca_full_corrected.cpk"

gp_bin5_path = MODEL_DIR + "gp_bin5.cpk"
pca_bin5_path = MODEL_DIR + "pca_bin5.cpk"
standardizer_bin5_path = MODEL_DIR + "standardizer_bin5.cpk"

k_path = MODEL_DIR + "cola_eg.txt"   # or wherever your k-grid is stored


# -------------------------------------------------
# Load emulators
# -------------------------------------------------
linear_emu = LinearBoostNN(linear_model_path)

gp_emu = NonlinearBoostGP(
    gp_full_path,
    pca_full_path,
    gp_bin5_path,
    pca_bin5_path,
    standardizer_bin5_path,
    k_path
)

emu = BoostEmulator(linear_emu, gp_emu)


# -------------------------------------------------
# Define cosmology + MG params
# -------------------------------------------------
cosmo = {
    "Omega_m": 0.31,
    "Omega_b": 0.049,
    "h": 0.67,
    "n_s": 0.96,
    "A_s": 2.1e-9
}

mu = 1.05
eta = 1.0
bin_index = 2

zs = [0.0, 0.5, 1.0]


# -------------------------------------------------
# Run emulator
# -------------------------------------------------
k, boost = emu.predict_boost(
    cosmo,
    mu=mu,
    eta=eta,
    bin_index=bin_index,
    zs=zs
)


# -------------------------------------------------
# Plot
# -------------------------------------------------
plt.figure(figsize=(6,4))

for i, z in enumerate(zs):
    plt.semilogx(k, boost[i], label=f"z = {z}")

plt.axhline(1.0, linestyle="--", color="black")

plt.xlabel("k [h/Mpc]")
plt.ylabel("Boost")
plt.title("MG Boost Emulator")

plt.legend()
plt.tight_layout()
plt.show()
