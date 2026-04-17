import numpy as np
import cloudpickle as cpk

class NonlinearBoostGP:

    def __init__(
        self,
        gp_full_path,
        pca_full_path,
        gp_bin5_path,
        pca_bin5_path,
        standardizer_bin5_path,
        k_path
    ):

        # -----------------------------
        # FULL GP (bins 0–3)
        # -----------------------------
        with open(pca_full_path, "rb") as f:
            self.pca_full = cpk.load(f)

        with open(gp_full_path, "rb") as f:
            self.gp_full = cpk.load(f)

        # -----------------------------
        # BIN 5 GP
        # -----------------------------
        with open(pca_bin5_path, "rb") as f:
            self.pca_bin5 = cpk.load(f)

        with open(gp_bin5_path, "rb") as f:
            self.gp_bin5 = cpk.load(f)

        with open(standardizer_bin5_path, "rb") as f:
            self.standardizer_bin5 = cpk.load(f)

        # -----------------------------
        # k-grid
        # -----------------------------
        self.k = np.loadtxt(k_path, usecols=0)

        # -----------------------------
        # bin remapping (for FULL GP only)
        # -----------------------------
        self.bin_map = {
            0: 0,
            1: 3,
            2: 2,
            3: 1,
            4: 4
        }


    # -------------------------------------------------
    # Prediction
    # -------------------------------------------------
    def predict_boost(self, cosmo, mu, bin_index, zs):

        zs = np.atleast_1d(zs)
        N = len(zs)

        bin_index = int(bin_index)

        # =================================================
        # BIN 5 (special case)
        # =================================================
        if bin_index == 4:

            X = np.column_stack([
                np.full(N, cosmo["Omega_m"]),
                np.full(N, cosmo["h"]),
                np.full(N, cosmo["Omega_b"]),
                np.full(N, cosmo["n_s"]),
                np.full(N, cosmo["A_s"]),
                np.full(N, mu),
                zs
            ])

            # --- standardize inputs ---
            X = self.standardizer_bin5.standardize(X)

            # --- GP prediction ---
            pca_modes = self.gp_bin5.predict(X)

            # --- PCA reconstruction ---
            boost = self.pca_bin5.inverse_transform(pca_modes)

            # --- exponentiate ---
            boost = np.exp(boost)

        # =================================================
        # FULL GP (bins 0–3)
        # =================================================
        else:

            gp_bin = self.bin_map[bin_index]

            X = np.column_stack([
                np.full(N, cosmo["Omega_m"]),
                np.full(N, cosmo["h"]),
                np.full(N, cosmo["Omega_b"]),
                np.full(N, cosmo["n_s"]),
                np.full(N, cosmo["A_s"]),
                np.full(N, mu),
                np.full(N, gp_bin),
                zs
            ])

            pca_modes = self.gp_full.predict(X)

            boost = self.pca_full.inverse_transform(pca_modes)

        return self.k, boost
