# %%
from MolDyn import MolDyn
from RDF import RDF
from Plots import Plots
import matplotlib.pyplot as plt
import numpy as np
import time

file_format = "png"
dpi = 400

temp = 94
thermo_rate = 10
box_scale = 1
equilibration = 500
Niter = 2000

# %%
"""
 RDF PLOT 
"""

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
fig, ax = plt.subplots()
ins_1.plot_rdf(label=r"$R_{cut}=1$", fmt="o-", fig=fig, ax=ax)
ins_2.plot_rdf(label=r"$R_{cut}=2$", fmt="s-", fig=fig, ax=ax)
ax.set_xlim(0, 6)

fig.savefig(
    f"rdf_combined_T{temp}_L{box_scale}_Tr{thermo_rate}_Eq{equilibration}_step{Niter}.{file_format}",
    bbox_inches="tight",
    dpi=dpi,
)

# %%
"""
CDF Plot
"""
fig, ax = plt.subplots()
ins_1.plot_cdf(label=r"$R_{cut}=1$", fig=fig, ax=ax, linestyle="--")
ins_2.plot_cdf(label=r"$R_{cut}=2$", fig=fig, ax=ax, linestyle="-")
fig.savefig(
    f"cdf_combined_T{temp}_L{box_scale}_Tr{thermo_rate}_Eq{equilibration}_step{Niter}.{file_format}",
    bbox_inches="tight",
    dpi=dpi,
)
# zoom-in at rcut and  highlight nearest neighbors
ax.set_xlim(0, ins_2.Rcut / ins_1.sigma)
ax.set_ylim(0, 400)

ax.axhline(12, 0, ins_2.Rcut / ins_1.sigma, color="black", linestyle=":", linewidth=1)
ax.axhline(
    12 + 6, 0, ins_2.Rcut / ins_1.sigma, color="black", linestyle=":", linewidth=1
)
ax.axhline(
    12 + 6 + 24, 0, ins_2.Rcut / ins_1.sigma, color="black", linestyle=":", linewidth=1
)
fig.savefig(
    f"cdf_combined_T{temp}_L{box_scale}_Tr{thermo_rate}_Eq{equilibration}_step{Niter}_zoom.{file_format}",
    bbox_inches="tight",
    dpi=dpi,
)

# %%
