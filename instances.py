# %%
from MolDyn import MolDyn
from RDF import RDF
from Plots import Plots
import matplotlib.pyplot as plt
import numpy as np
import time

file_format = "png"
dpi = 400

temp = 50
thermo_rate = 10
box_scale = 2
rcut_scale = 1
equilibration = 500
Niter = 2000
# %%
"""
RUN MOLECULAR DYNAMICS
"""

ins = MolDyn(
    temp=temp,
    thermo_rate=thermo_rate,
    box_scale=box_scale,
    rcut_scale=rcut_scale,
    equilibration=equilibration,
    Niter=Niter,
)
# time the execution
start = time.time()
# ins.run_md()
end = time.time()
print("execution time: ", end - start)


# %%
"""
 TEMPERATURE PLOT 
"""

ins = Plots(
    temp=temp,
    box_scale=box_scale,
    rcut_scale=rcut_scale,
    thermo_rate=thermo_rate,
    equilibration=equilibration,
    Niter=Niter,
)

fig, ax = plt.subplots()
ins.plot_temp(start_time=500, fig=fig, ax=ax)

fig.savefig(
    f"plots/temperatures_T{temp}_L{box_scale}_Rc{rcut_scale}_Tr{thermo_rate}_Eq{equilibration}_step{Niter}.{file_format}",
    bbox_inches="tight",
    dpi=dpi,
)
# %% temp rms deviation
temp_dat = ins.temp_arr[equilibration : ins.Niter]
np.sqrt(np.mean(temp_dat**2) - np.mean(temp_dat) ** 2) / np.mean(temp_dat)

# %%
"""
 RDF specific temp PLOT 
"""

ins = Plots(
    temp=temp,
    box_scale=box_scale,
    rcut_scale=rcut_scale,
    thermo_rate=thermo_rate,
    equilibration=equilibration,
    Niter=Niter,
)
fig, ax = plt.subplots()
ins.plot_rdf(label=f"T={temp} K", fmt="o-", fig=fig, ax=ax)
ax.set_title(
    rf"T={temp} K, L={box_scale}, R$_{{cut}}$={rcut_scale}",
    fontsize=20,
    fontweight="bold",
)
fig.savefig(
    f"plots/rdf_T{temp}_L{box_scale}_Rc{rcut_scale}_Tr{thermo_rate}_Eq{equilibration}_step{Niter}.{file_format}",
    bbox_inches="tight",
    dpi=dpi,
)


# %%
"""
 RDF PLOT 
"""

ins_50 = Plots(
    temp=50,
    box_scale=box_scale,
    rcut_scale=rcut_scale,
    thermo_rate=thermo_rate,
    equilibration=equilibration,
    Niter=Niter,
)

ins_94 = Plots(
    temp=94,
    box_scale=box_scale,
    rcut_scale=rcut_scale,
    thermo_rate=thermo_rate,
    equilibration=equilibration,
    Niter=Niter,
)

ins_300 = Plots(
    temp=300,
    box_scale=box_scale,
    rcut_scale=rcut_scale,
    thermo_rate=thermo_rate,
    equilibration=equilibration,
    Niter=Niter,
)


# %%
fig, ax = plt.subplots()
ins_50.plot_specific_rdf(
    time_snapshot=0, label=r"initial", linestyle=":", fig=fig, ax=ax
)
ins_50.plot_rdf(label="T=50 K", fmt="o-", fig=fig, ax=ax)
ins_94.plot_rdf(label="T=94 K", fmt="s-", fig=fig, ax=ax)
ins_300.plot_rdf(label="T=300 K", fmt="^-", fig=fig, ax=ax)
ax.set_title(
    rf"L={box_scale}, R$_{{cut}}$={rcut_scale}",
    fontsize=20,
    fontweight="bold",
)
fig.savefig(
    f"plots/rdf_combined_L{box_scale}_Rc{rcut_scale}_Tr{thermo_rate}_Eq{equilibration}_step{Niter}.{file_format}",
    bbox_inches="tight",
    dpi=dpi,
)
# %%
"""
CDF Plot
"""
fig, ax = plt.subplots()
ins_50.plot_specific_cdf(
    time_snapshot=0, label=r"initial", fig=fig, ax=ax, linestyle=":"
)
ins_50.plot_cdf(label=f"T=50 K", fig=fig, ax=ax, linestyle="--")
ins_94.plot_cdf(label=f"T=94 K", fig=fig, ax=ax, linestyle="-")
ins_300.plot_cdf(label=f"T=300 K", fig=fig, ax=ax, linestyle="-.")

ax.set_title(
    rf"L={box_scale}, R$_{{cut}}$={rcut_scale}",
    fontsize=20,
    fontweight="bold",
)
fig.savefig(
    f"plots/cdf_combined_L{box_scale}_Rc{rcut_scale}_Tr{thermo_rate}_Eq{equilibration}_step{Niter}.{file_format}",
    bbox_inches="tight",
    dpi=dpi,
)
# zoom-in at rcut and  highlight nearest neighbors
ax.set_xlim(0, ins_50.Rcut / ins_50.sigma)
ax.set_ylim(0, 60)

fig.savefig(
    f"plots/cdf_combined_L{box_scale}_Rc{rcut_scale}_Tr{thermo_rate}_Eq{equilibration}_step{Niter}_zoom.{file_format}",
    bbox_inches="tight",
    dpi=dpi,
)


# %%
# %matplotlib qt
"""
 ANIMATION
"""

ins = Plots(
    temp=temp,
    box_scale=box_scale,
    rcut_scale=rcut_scale,
    thermo_rate=thermo_rate,
    equilibration=equilibration,
    Niter=Niter,
)
folder = "/home/loerl/ictp-diploma/num2/project/tmp"
ins.make_animation(folder)
