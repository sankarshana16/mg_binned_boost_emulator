import numpy as np
from .utils import check_param_ranges



class BoostEmulator:

    def __init__(
    self,
    linear_emu,
    nonlinear_emu,
    k_low=1e-2,
    k_high=1e-1):

        self.linear = linear_emu
        self.nonlinear = nonlinear_emu

        self.k_low = k_low
        self.k_high = k_high

        # fixed output grid
        self.k_lin = self.linear.k

        # fixed GP grid
        self.k_gp = self.nonlinear.k

        # precompute blending weights once
        self.w = self._compute_weights(self.k_lin)[None, :]

    # -------------------------------------------------
    # Compute blending weights
    # -------------------------------------------------
    def _compute_weights(self, k):

        logk = np.log(k)

        w = (logk - np.log(self.k_low)) / (
            np.log(self.k_high) - np.log(self.k_low)
        )

        w = np.clip(w, 0.0, 1.0)

        return w


    # -------------------------------------------------
    # Main prediction
    # -------------------------------------------------
    def predict_boost(self, cosmo, mu, eta, bin_index, zs):

        zs = np.atleast_1d(zs)

        # optional validation only if desired
        check_param_ranges(cosmo, mu, eta, bin_index, zs)

        # Linear boost
        _, boost_lin = self.linear.predict_boost(
            cosmo, mu, eta, bin_index, zs
        )

        # Nonlinear boost
        _, boost_gp = self.nonlinear.predict_boost(
            cosmo, mu, bin_index, zs
        )

        # Fast interpolation GP → NN grid
        boost_gp_interp = np.array([
            np.interp(
                self.k_lin,
                self.k_gp,
                row
            )
            for row in boost_gp
        ])

        # Blend
        boost = (1 - self.w) * boost_lin + self.w * boost_gp_interp

        return self.k_lin, boost
