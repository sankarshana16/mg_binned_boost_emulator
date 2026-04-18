from .linear_nn import LinearBoostNN
from .gp_emulator import NonlinearBoostGP
from .stitching import BoostEmulator


class MGEmulator:

    def __init__(self, model_dir="models/"):

        # -----------------------------
        # Linear emulator
        # -----------------------------
        linear_emu = LinearBoostNN(
            f"{model_dir}/linear_boost_nn.pt"
        )

        # -----------------------------
        # Nonlinear emulator
        # -----------------------------
        gp_emu = NonlinearBoostGP(
            f"{model_dir}/gp_full_corrected.cpk",
            f"{model_dir}/pca_full_corrected.cpk",
            f"{model_dir}/gp_bin5.cpk",
            f"{model_dir}/pca_bin5.cpk",
            f"{model_dir}/standardizer_bin5.cpk",
            f"{model_dir}/cola_eg.txt"
        )

        # -----------------------------
        # Final stitched emulator
        # -----------------------------
        self.emulator = BoostEmulator(
            linear_emu,
            gp_emu
        )

    def predict_boost(
        self,
        cosmo,
        mu,
        eta,
        bin_index,
        zs
    ):

        return self.emulator.predict_boost(
            cosmo,
            mu,
            eta,
            bin_index,
            zs
        )

    def __call__(
        self,
        cosmo,
        mu,
        eta,
        bin_index,
        zs
    ):
        """
        Optional shorthand:
        k, boost = emu(...)
        """

        return self.predict_boost(
            cosmo,
            mu,
            eta,
            bin_index,
            zs
        )
