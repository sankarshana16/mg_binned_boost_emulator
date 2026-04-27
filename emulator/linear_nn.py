import numpy as np
from scipy.interpolate import interp1d
import cloudpickle as cpk
import matplotlib.pyplot as plt
import torch
from emulator.model import Net


class LinearBoostNN:

    def __init__(self, model_path, device="cpu"):

        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        self.model = Net(
            hidden_layers=checkpoint["hidden_layers"],
            output_dim=checkpoint["output_dim"]
        ).to(device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.device = device

        # normalization
        self.mu_param = checkpoint["mu_param"].to(device)
        self.sigma_param = checkpoint["sigma_param"].to(device)

        self.y_mu = checkpoint["y_mu"].cpu().numpy()
        self.y_sigma = checkpoint["y_sigma"].cpu().numpy()

        self.k = checkpoint["k"]

    def _map_params(self, cosmo, mu, eta, bin_index, z):

        lnAs = np.log(1e10 * cosmo["A_s"])
        bin_scaled = bin_index / 4.0 

        return np.array([
            cosmo["Omega_m"],
            cosmo["Omega_b"],
            cosmo["h"],
            cosmo["n_s"],
            lnAs,
            mu,
            eta,
            z,
            bin_scaled
        ], dtype=np.float32)

    def _standardize(self, x):
        return (x - self.mu_param) / self.sigma_param

    def predict_boost(self, cosmo, mu, eta, bin_index, zs):

        zs = np.atleast_1d(zs).astype(np.float32)
        Nz = len(zs)

        lnAs = np.log(1e10 * cosmo["A_s"]).astype(np.float32)
        bin_scaled = np.float32(bin_index / 4.0)

        # Build full batch at once: shape (Nz, 9)
        X = np.column_stack([
            np.full(Nz, cosmo["Omega_m"], dtype=np.float32),
            np.full(Nz, cosmo["Omega_b"], dtype=np.float32),
            np.full(Nz, cosmo["h"], dtype=np.float32),
            np.full(Nz, cosmo["n_s"], dtype=np.float32),
            np.full(Nz, lnAs, dtype=np.float32),
            np.full(Nz, mu, dtype=np.float32),
            np.full(Nz, eta, dtype=np.float32),
            zs,
            np.full(Nz, bin_scaled, dtype=np.float32),
        ])

        # One tensor conversion only
        x = torch.from_numpy(X).to(self.device)

        # Vectorized normalization
        x = (x - self.mu_param) / self.sigma_param

        with torch.no_grad():
            pred_std = self.model(x).cpu().numpy()

        pred = pred_std * self.y_sigma + self.y_mu

        shape = pred[:, :-1]
        log_ref = pred[:, -1:]
    
        log_boost = shape + log_ref
        boost = np.exp(log_boost)
    
        return self.k, boost
