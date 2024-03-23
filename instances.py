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
rcut_scale = 2
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
ins.run_md()
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
ins_50.plot_rdf(label="T=50 K", fmt="o-", fig=fig, ax=ax)
ins_94.plot_rdf(label="T=94 K", fmt="s-", fig=fig, ax=ax)
ins_300.plot_rdf(label="T=300 K", fmt="^-", fig=fig, ax=ax)
fig.savefig(
    f"plots/rdf_combined_L{box_scale}_Rc{rcut_scale}_Tr{thermo_rate}_Eq{equilibration}_step{Niter}.{file_format}",
    bbox_inches="tight",
    dpi=dpi,
)
# ax.set_xlim(0,8)
# %%
"""
CDF Plot
"""
fig, ax = plt.subplots()
ins_50.plot_cdf(label=f"T=50 K", fig=fig, ax=ax, linestyle="--")
ins_94.plot_cdf(label=f"T=94 K", fig=fig, ax=ax, linestyle="-")
ins_300.plot_cdf(label=f"T=300 K", fig=fig, ax=ax, linestyle=":")
fig.savefig(
    f"plots/cdf_combined_L{box_scale}_Rc{rcut_scale}_Tr{thermo_rate}_Eq{equilibration}_step{Niter}.{file_format}",
    bbox_inches="tight",
    dpi=dpi,
)
# zoom-in at rcut and  highlight nearest neighbors
ax.set_xlim(0, ins_50.Rcut / ins.sigma)
ax.set_ylim(0, 60)

ax.axhline(12, 0, ins_50.Rcut / ins.sigma, color="black", linestyle=":", linewidth=1)
ax.axhline(
    12 + 6, 0, ins_50.Rcut / ins.sigma, color="black", linestyle=":", linewidth=1
)
ax.axhline(
    12 + 6 + 24, 0, ins_50.Rcut / ins.sigma, color="black", linestyle=":", linewidth=1
)

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

# ax = plt.figure().add_subplot(projection="3d")
# plt.subplots_adjust(right=1, top=1, left=0, bottom=0)
# for t in range(0, ins.Niter):
#     ax.clear()
#     ax.set_xlim(0, ins.L)
#     ax.set_ylim(0, ins.L)
#     ax.set_zlim(0, ins.L)
#     ax.scatter(
#         ins.pos_config[t, :, 0][:],
#         ins.pos_config[t, :, 1][:],
#         ins.pos_config[t, :, 2][:],
#     )
# plt.pause(0.03)
# plt.savefig(f"{folder}/{t}.png", dpi=100)
# plt.show()


# %%
# single rdf
# pos = ins.pos_config[-1]
# i = ins.N // 2
# pos_diff = pos[i] - pos
# # remove self interaction (j=i)
# pos_diff = np.delete(pos_diff, i, 0)
# pos_diff = ins.min_image(pos_diff)
# dist = np.linalg.norm(
#     pos_diff, ord=2, axis=1
# )  # distance between ith particle and others

# cut up to L/2
# dist = dist[dist <= ins.L / 2]

# fig, ax = plt.subplots()
# h, bin_edges, patches = ax.hist(
#     dist,
#     density=0,
#     bins=30,
#     linewidth=2.0,
# )
# h, bin_edges = np.histogram(
#     dist,
#     density=0,
#     bins=30,
# )
# dr = bin_edges[1] - bin_edges[0]
# bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2.0
# denominator = 4 * np.pi * bin_centers**2 * dr * ins.N / ins.L**3
# rdf = h / denominator

# ax.vlines(ins.L / 2, 0, 60, color="red")
# fig, ax = plt.subplots()
# ax.plot(bin_centers, rdf, marker="o")
