# %%
from MolDyn import MolDyn
from RDF import RDF
from Plots import Plots
import matplotlib.pyplot as plt

# %%
"""
RUN MOLECULAR DYNAMICS
"""

ins = MolDyn(
    temp=50,
    box_scale=1,
    thermo_rate=10,
    equilibration=500,
    Niter=1000,
)
ins.run_md()

# %%
"""
 TEMPERATURE PLOT 
"""
temp = 50
box_scale = 1
thermo_rate = 10
equilibration = 500
Niter = 1000

ins = Plots(
    temp=temp,
    box_scale=box_scale,
    thermo_rate=thermo_rate,
    equilibration=equilibration,
    Niter=Niter,
)

fig, ax = plt.subplots()
ins.plot_temp(start_time=200, fig=fig, ax=ax)

# %%
"""
 RDF specific temp PLOT 
"""
temp = 50
box_scale = 1
thermo_rate = 10
equilibration = 500
Niter = 1000


ins = Plots(
    temp=temp,
    box_scale=box_scale,
    thermo_rate=thermo_rate,
    equilibration=equilibration,
    Niter=Niter,
)
fig, ax = plt.subplots()
ins.plot_rdf(label=f"T={temp} K", fmt="o-", fig=fig, ax=ax)
fig.savefig(f"rdf_T{temp}_L{box_scale}.pdf", bbox_inches="tight")

# %%
"""
 RDF PLOT 
"""
box_scale = 1
thermo_rate = 10
equilibration = 500
Niter = 2000

ins_50 = Plots(
    temp=50,
    box_scale=box_scale,
    thermo_rate=thermo_rate,
    equilibration=equilibration,
    Niter=Niter,
)

ins_94 = Plots(
    temp=94,
    box_scale=box_scale,
    thermo_rate=thermo_rate,
    equilibration=equilibration,
    Niter=Niter,
)

ins_300 = Plots(
    temp=300,
    box_scale=box_scale,
    thermo_rate=thermo_rate,
    equilibration=equilibration,
    Niter=Niter,
)

fig, ax = plt.subplots()
ins_50.plot_rdf(label="T=50 K", fmt="o-", fig=fig, ax=ax)
ins_94.plot_rdf(label="T=94 K", fmt="s-", fig=fig, ax=ax)
ins_300.plot_rdf(label="T=300 K", fmt="^-", fig=fig, ax=ax)
fig.savefig(f"rdf_combined_L{box_scale}.pdf", bbox_inches="tight")

# %%
# ins_50 = RDF(temp=50, box_scale=box_scale)
# ins_50.run_md()
# ins_94 = RDF(temp=94, box_scale=box_scale)
# ins_94.run_md()
# ins_400 = RDF(temp=400, box_scale=box_scale)
# ins_400.run_md()

# %%
# fig, ax = plt.subplots()

# x_vals, avg, errors = ins_50.combine_rdf()
# ax.errorbar(
#     x_vals, avg, yerr=errors, markersize=4, fmt="o-", capsize=3.5, label="T=50 K"
# )
# x_vals, avg, errors = ins_94.combine_rdf()
# ax.errorbar(
#     x_vals, avg, yerr=errors, markersize=4, fmt="s-", capsize=3.5, label="T=94 K"
# )
# x_vals, avg, errors = ins_400.combine_rdf()
# ax.errorbar(
#     x_vals, avg, yerr=errors, markersize=4, fmt="^-", capsize=3.5, label="T=400 K"
# )
# ax.set_ylabel("radial distribution function", fontsize=15)
# ax.set_xlabel(r"r/$\sigma$", fontsize=15)
# ax.set_xlim(0, 5)
# ax.set_ylim(0)
# ax.legend()

# fig.savefig(f"rdf_combined.pdf", bbox_inches="tight")

# %%
# %matplotlib qt
"""
 ANIMATION
"""

temp = 50
box_scale = 1
thermo_rate = 10
equilibration = 500
Niter = 1000

ins = Plots(
    temp=temp,
    box_scale=box_scale,
    thermo_rate=thermo_rate,
    equilibration=equilibration,
    Niter=Niter,
)
folder = "/home/loerl/ictp-diploma/num2/project/tmp"
# ins.load_pos_config()
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
# import glob
# from PIL import Image
# import imageio

# folder = "/home/loerl/ictp-diploma/num2/project/tmp"


# def make_gif(frame_folder):
#     frames = [Image.open(image) for image in sorted(glob.glob(f"{frame_folder}/*.png"))]
#     frame_one = frames[0]
#     frame_one.save(
#         "my_awesome.gif",
#         format="GIF",
#         append_images=frames,
#         save_all=True,
#         duration=1,
#         loop=1,
#     )


# make_gif(folder)


# %% temp equilibrium
ins = ins_94
plt.plot(ins.temp_arr[00:])
plt.hlines(np.mean(ins.temp_arr[200:]), 0, ins.Niter, linestyles="dotted")
np.mean(ins.temp_arr[150:])
plt.xlabel(r"timestep (t/$\Delta$t)")
plt.ylabel("Temperature (K)")
plt.savefig(f"temp_eq_10rate.pdf", bbox_inches="tight")

# %% temp rms deviation
temp_dat = ins.temp_arr[200 : ins.Niter]
np.sqrt(np.mean(temp_dat**2) - np.mean(temp_dat) ** 2) / np.mean(temp_dat)

# %%
# rdf
pos = ins.pos_config[-1]
i = ins.N // 2
pos_diff = pos[i] - pos
# remove self interaction (j=i)
pos_diff = np.delete(pos_diff, i, 0)
pos_diff = ins.min_image(pos_diff)
dist = np.linalg.norm(
    pos_diff, ord=2, axis=1
)  # distance between ith particle and others

# cut up to L/2
dist = dist[dist <= ins.L / 2]

fig, ax = plt.subplots()
h, bin_edges, patches = ax.hist(
    dist,
    density=0,
    bins=30,
    # histtype="step",
    # label="T=100 K",
    # color=colors[0],
    linewidth=2.0,
)
h, bin_edges = np.histogram(
    dist,
    density=0,
    bins=30,
)
dr = bin_edges[1] - bin_edges[0]
bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2.0
denominator = 4 * np.pi * bin_centers**2 * dr * ins.N / ins.L**3
rdf = h / denominator

ax.vlines(ins.L / 2, 0, 60, color="red")
# %%

fig, ax = plt.subplots()
ax.plot(bin_centers, rdf, marker="o")


# %%
