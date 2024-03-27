# %%
from MolDyn import MolDyn
import numpy as np
import matplotlib.pyplot as plt


class MSD(MolDyn):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_msd(self):
        timesteps = range(1, (self.Niter - self.equilibration) // 4)
        data = self.pos_config[self.equilibration :, :, :]
        self.dt_arr = np.empty(len(timesteps))
        self.msd_arr = np.empty(len(timesteps))
        self.msd_error_arr = np.empty(len(timesteps))

        for count, dt in enumerate(timesteps):
            deltaCoords = data[dt:, :, :] - data[0:-dt, :, :]
            deltaCoords = self.min_image(deltaCoords)
            squaredDisplacement = np.linalg.norm(deltaCoords, ord=2, axis=2) ** 2
            # take ensemble average
            msd_ensemble_avg = np.mean(squaredDisplacement, axis=1)
            # take time average
            self.msd_arr[count] = np.mean(msd_ensemble_avg) / 1e-20  # in angstrom
            self.dt_arr[count] = dt * 1e-14 / 1e-12  # in picosecond
            self.msd_error_arr[count] = np.std(msd_ensemble_avg, ddof=1)

    def fit_msd(self, start=250):
        self.msd_coef = np.polyfit(self.dt_arr[start:], self.msd_arr[start:], 1)
        return np.poly1d(self.msd_coef)
