# %%
from MolDyn import MolDyn
from RDF import RDF
from Plots import Plots
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

colors = sns.color_palette("deep")

file_format = "png"
dpi = 400

temp = 94
thermo_rate = 10
box_scale = 1
equilibration = 500
Niter = 2000

ins_1 = Plots(
    temp=temp,
    box_scale=box_scale,
    rcut_scale=1,
    thermo_rate=thermo_rate,
    equilibration=equilibration,
    Niter=Niter,
)

ins_2 = Plots(
    temp=temp,
    box_scale=box_scale,
    rcut_scale=2,
    thermo_rate=thermo_rate,
    equilibration=equilibration,
    Niter=Niter,
)

# %%
"""
 Velocity Distribution PLOT 
"""
for keyword in ["initial", "final"]:
    if keyword == "initial":
        time_snapshot = 0
        show_exact = False
    elif keyword == "final":
        time_snapshot = -1
        show_exact = True

    fig, ax = plt.subplots()
    ins_1.plot_vel_dist(
        fig,
        ax,
        label=rf"$R_{{cut}}={ins_1.rcut_scale}$",
        color=colors[0],
        time_snapshot=time_snapshot,
        show_exact=show_exact,
    )
    ins_2.plot_vel_dist(
        fig,
        ax,
        label=rf"$R_{{cut}}={ins_2.rcut_scale}$",
        color=colors[1],
        time_snapshot=time_snapshot,
        show_exact=show_exact,
    )
    ax.set_title(
        rf"Velocity Distribution at T={temp} K, L={box_scale}",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(
        f"plots/veldist_T{temp}_L{box_scale}_Tr{thermo_rate}_Eq{equilibration}_step{Niter}_{keyword}.{file_format}",
        bbox_inches="tight",
        dpi=dpi,
    )
# %%
"""
 RDF PLOT 
"""

fig, ax = plt.subplots()
ins_1.plot_specific_rdf(
    time_snapshot=0, label=r"initial", linestyle=":", fig=fig, ax=ax
)
ins_1.plot_rdf(label=rf"$R_{{cut}}={ins_1.rcut_scale}$", fmt="o-", fig=fig, ax=ax)
ins_2.plot_rdf(label=rf"$R_{{cut}}={ins_2.rcut_scale}$", fmt="s-", fig=fig, ax=ax)
ax.set_title(
    rf"T={temp} K, L={box_scale}",
    fontsize=20,
    fontweight="bold",
)
fig.savefig(
    f"plots/rdf_combined_T{temp}_L{box_scale}_Tr{thermo_rate}_Eq{equilibration}_step{Niter}.{file_format}",
    bbox_inches="tight",
    dpi=dpi,
)

# %%
"""
CDF Plot
"""
fig, ax = plt.subplots()
ins_1.plot_specific_cdf(
    time_snapshot=0, label=r"initial", fig=fig, ax=ax, linestyle=":"
)
ins_1.plot_cdf(label=rf"$R_{{cut}}={ins_1.rcut_scale}$", fig=fig, ax=ax, linestyle="--")
ins_2.plot_cdf(label=rf"$R_{{cut}}={ins_2.rcut_scale}$", fig=fig, ax=ax, linestyle="-")
ax.set_title(
    rf"T={temp} K, L={box_scale}",
    fontsize=20,
    fontweight="bold",
)
fig.savefig(
    f"plots/cdf_combined_T{temp}_L{box_scale}_Tr{thermo_rate}_Eq{equilibration}_step{Niter}.{file_format}",
    bbox_inches="tight",
    dpi=dpi,
)
# zoom-in at rcut and  highlight nearest neighbors
ax.set_xlim(0, ins_2.Rcut / ins_1.sigma)
ax.set_ylim(0, 400)


fig.savefig(
    f"plots/cdf_combined_T{temp}_L{box_scale}_Tr{thermo_rate}_Eq{equilibration}_step{Niter}_zoom.{file_format}",
    bbox_inches="tight",
    dpi=dpi,
)

# %%
