# %%
from MolDyn import MolDyn
from RDF import RDF
from Plots import Plots
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time

colors = sns.color_palette("deep")

file_format = "png"
dpi = 400

temp = 300
box_scale = 2
rcut_scale = 1
thermo_rate = 10
equilibration = 500
Niter = 2000

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
## time the execution
start = time.time()
# ins.run_md()
end = time.time()
print("execution time: ", end - start)

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
    ins_50.plot_vel_dist(
        fig,
        ax,
        label=f"T={ins_50.Temp} K",
        color=colors[0],
        time_snapshot=time_snapshot,
        show_exact=show_exact,
    )
    ins_94.plot_vel_dist(
        fig,
        ax,
        label=f"T={ins_94.Temp} K",
        color=colors[1],
        time_snapshot=time_snapshot,
        show_exact=show_exact,
    )
    ins_300.plot_vel_dist(
        fig,
        ax,
        label=f"T={ins_300.Temp} K",
        color=colors[2],
        time_snapshot=time_snapshot,
        show_exact=show_exact,
    )
    ax.set_title(
        rf"Velocity Distribution at L={box_scale}, R$_{{cut}}$={rcut_scale}",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(
        f"plots/veldist_L{box_scale}_Rc{rcut_scale}_Tr{thermo_rate}_Eq{equilibration}_step{Niter}_{keyword}.{file_format}",
        bbox_inches="tight",
        dpi=dpi,
    )

# %%
"""
 Speed Distribution PLOT 
"""
s = np.linspace(0, 1200, 1000)
vel_norm = np.linalg.norm(ins_50.vel_config[-1], ord=2, axis=1)
fig, ax = plt.subplots()
ax.hist(vel_norm, density=1)
ax.plot(s, ins_50.speed_MB(s))
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

fig, ax = plt.subplots()
ins_50.plot_specific_rdf(
    time_snapshot=0, label=r"initial", linestyle=":", fig=fig, ax=ax
)
ins_50.plot_rdf(label=f"T={ins_50.Temp} K", fmt="o-", fig=fig, ax=ax)
ins_94.plot_rdf(label=f"T={ins_94.Temp} K", fmt="s-", fig=fig, ax=ax)
ins_300.plot_rdf(label=f"T={ins_300.Temp} K", fmt="^-", fig=fig, ax=ax)
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
ins_50.plot_cdf(label=f"T={ins_50.Temp} K", fig=fig, ax=ax, linestyle="--")
ins_94.plot_cdf(label=f"T={ins_94.Temp} K", fig=fig, ax=ax, linestyle="-")
ins_300.plot_cdf(label=f"T={ins_300.Temp} K", fig=fig, ax=ax, linestyle="-.")

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

# %%
