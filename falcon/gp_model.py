from __future__ import annotations

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel


class ComponentGP:
    def __init__(self, length_scale=0.75, signal_variance=4.0,
                 noise_variance=0.015**2, random_state=0):
        kernel = (
            ConstantKernel(signal_variance, constant_value_bounds="fixed")
            * Matern(length_scale=length_scale,
                     nu=2.5, length_scale_bounds="fixed")
            + WhiteKernel(noise_level=noise_variance,
                          noise_level_bounds="fixed")
        )
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            optimizer=None,
            normalize_y=False,
            random_state=random_state,
        )
        self.x = None
        self.y = None

    def fit(self, t, y):
        t = np.asarray(t, dtype=float).reshape(-1, 1)
        y = np.asarray(y, dtype=float)
        self.x = t[:, 0]
        self.y = y
        self.model.fit(t, y)

    def predict(self, t_grid, return_std=True):
        t_grid = np.asarray(t_grid, dtype=float).reshape(-1, 1)
        return self.model.predict(t_grid, return_std=return_std)

    def sample_trajectories(self, t_grid, n_samples=50, rng_seed=0):
        t_grid = np.asarray(t_grid, dtype=float).reshape(-1, 1)
        samples = self.model.sample_y(
            t_grid, n_samples=n_samples, random_state=rng_seed
        )
        if samples.ndim == 1:
            samples = samples[:, None]
        return samples.T
