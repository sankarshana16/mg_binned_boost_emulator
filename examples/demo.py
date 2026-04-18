import numpy as np
import matplotlib.pyplot as plt

from emulator import MGEmulator



# -------------------------------------------------
# Load emulator
# -------------------------------------------------

emu = MGEmulator(model_dir="./models")


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

mus = [0.9, 1.1]  # The extreme values

colors = {
    0.9: "blue",
    1.1: "red"
}

linestyles = {
    0.0: "-",
    0.5: "--",
    1.0: ":"
}

zs = [0.0, 0.5, 1.0]

plt.figure(figsize=(6,4))

for m in mus:

    k, boost = emu.predict_boost(
        cosmo,
        mu=m,
        eta=1.0,
        bin_index=1,
        zs=zs
    )

    for i, z in enumerate(zs):
        plt.semilogx(
            k,
            boost[i],
            color=colors[m],
            linestyle=linestyles[z],
            label=rf"$\mu={m},\, z={z}$"
        )

plt.axhline(1.0, linestyle="--", color="black", alpha=0.6)

plt.xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$", fontsize=14)
plt.ylabel("Boost", fontsize=14)
plt.title(r"Effect of $\mu$ modified at $0.43 \leq z \leq 0.91$", fontsize=16)

plt.ylim(0.9, 1.1)

plt.legend(ncol=2, fontsize=10)
plt.tight_layout()
plt.savefig("./figures/boost_plot.png")
plt.show()
