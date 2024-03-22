# %%
from RDF import RDF
import numpy as np
import matplotlib.pyplot as plt
import glob
from PIL import Image


class Plots(RDF):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_temp_arr(self):
        self.temp_arr = np.load(
            f"data/temperatures_T{self.Temp}_L{self.box_scale}_Tr{self.thermo_rate}_Eq{self.equilibration}_step{self.Niter}.npy"
        )

    def load_pos_config(self):
        self.pos_config = np.load(
            f"data/positions_T{self.Temp}_L{self.box_scale}_Tr{self.thermo_rate}_Eq{self.equilibration}_step{self.Niter}.npy",
        )

    def plot_temp(self, start_time, fig, ax):
        self.load_temp_arr()
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
        self.load_pos_config()
        x_vals, avg, errors = self.combine_rdf()
        ax.errorbar(
            x_vals, avg, yerr=errors, markersize=4, fmt=fmt, capsize=3.5, label=label
        )
        ax.set_ylabel("radial distribution function", fontsize=15)
        ax.set_xlabel(r"r/$\sigma$", fontsize=15)
        ax.set_xlim(0, 5)
        ax.set_ylim(0)
        ax.legend()

    def make_animation(self, frame_folder):
        self.load_pos_config()
        ax = plt.figure().add_subplot(projection="3d")
        plt.subplots_adjust(right=1, top=1, left=0, bottom=0)
        for t in range(0, self.Niter, 20):
            ax.clear()
            ax.set_xlabel(r"x ($\AA$)")
            ax.set_ylabel(r"y ($\AA$)")
            ax.set_zlabel(r"z ($\AA$)")
            ax.set_xlim(0, self.L / 1e-10)
            ax.set_ylim(0, self.L / 1e-10)
            ax.set_zlim(0, self.L / 1e-10)
            ax.scatter(
                self.pos_config[t, :, 0] / 1e-10,
                self.pos_config[t, :, 1] / 1e-10,
                self.pos_config[t, :, 2] / 1e-10,
            )
            plt.savefig(f"{frame_folder}/{t}.png", dpi=100)

        frames = [
            Image.open(image) for image in sorted(glob.glob(f"{frame_folder}/*.png"))
        ]
        frame_one = frames[0]
        frame_one.save(
            f"animate_T{self.Temp}_L{self.box_scale}.gif",
            format="GIF",
            append_images=frames,
            save_all=True,
            duration=250,
            loop=1,
        )
