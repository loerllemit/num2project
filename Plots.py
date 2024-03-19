# %%
from RDF import RDF
import numpy as np
import matplotlib.pyplot as plt


class Plots(RDF):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def plot_temp(self, start_time, fig, ax):
        self.temp_arr = np.load(
            f"data/temperatures_T{self.Temp}_L{self.box_scale}_Tr{self.thermo_rate}_Eq{self.equilibration}_step{self.Niter}.npy"
        )

        ax.plot(range(start_time, self.Niter), self.temp_arr[start_time:])
        plt.hlines(
            np.mean(self.temp_arr[start_time:]),
            start_time,
            self.Niter,
            linestyles="dotted",
        )
        np.mean(self.temp_arr[start_time:])
        ax.set_xlabel(r"timestep (t/$\Delta$t)")
        ax.set_ylabel("Temperature (K)")
        fig.savefig(
            f"temperatures_T{self.Temp}_L{self.box_scale}_Tr{self.thermo_rate}_Eq{self.equilibration}_step{self.Niter}.pdf",
            bbox_inches="tight",
        )

    def plot_rdf(
        self,
        label,
        fig,
        ax,
        fmt="o-",
    ):
        self.pos_config = np.load(
            f"data/positions_T{self.Temp}_L{self.box_scale}_Tr{self.thermo_rate}_Eq{self.equilibration}_step{self.Niter}.npy",
        )
        x_vals, avg, errors = self.combine_rdf()
        ax.errorbar(
            x_vals, avg, yerr=errors, markersize=4, fmt=fmt, capsize=3.5, label=label
        )
        ax.set_ylabel("radial distribution function", fontsize=15)
        ax.set_xlabel(r"r/$\sigma$", fontsize=15)
        ax.set_xlim(0, 5)
        ax.set_ylim(0)
        ax.legend()
