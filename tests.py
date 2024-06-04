# %%
from Plots import Plots
import numpy as np
import matplotlib.pyplot as plt

"""
change these parameters
"""
temp = 94
box_scale = 1
rcut_scale = 1
thermo_rate = 10
equilibration = 500
Niter = 2000
#######################


ins = Plots(
    temp=temp,
    thermo_rate=thermo_rate,
    box_scale=box_scale,
    rcut_scale=rcut_scale,
    equilibration=equilibration,
    Niter=Niter,
)

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
data = ins.pos_config[ins.equilibration :, :, :]


# %%
dt_list = []
msd_list = []
error_list = []
for dt in range(1, (ins.Niter - ins.equilibration) // 4):
    deltaCoords = data[dt:, :, :] - data[0:-dt, :, :]
    deltaCoords = ins.min_image(deltaCoords)
    squaredDisplacement = np.linalg.norm(deltaCoords, ord=2, axis=2) ** 2
    msd_ensemble_avg = np.mean(squaredDisplacement, axis=1)
    msd_list.append(np.mean(msd_ensemble_avg))
    dt_list.append(dt)
    error_list.append(
        np.std(msd_ensemble_avg, ddof=1)
    )  # / np.sqrt(len(msd_ensemble_avg)))
# %%
msd_arr = np.array(msd_list) / 1e-20
dt_arr = np.array(dt_list)
coef = np.polyfit(dt_arr[250:], msd_arr[250:], 1)

poly1d_fn = np.poly1d(coef)

fig, ax = plt.subplots()
ax.plot(dt_arr, poly1d_fn(dt_arr))
ax.errorbar(
    dt_arr,
    msd_arr,
    yerr=error_list,
    # markersize=3,
    capsize=3.5,
)
# ax.set_xlim(0, 300)
ax.set_ylim(0)
ax.set_xlim(0)


# %%
