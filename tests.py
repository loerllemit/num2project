# %%
from MolDyn import MolDyn
from RDF import RDF
from Plots import Plots
import matplotlib.pyplot as plt
import numpy as np

# %%
"""
 TEST CDF 
"""
temp = 300
box_scale = 1
thermo_rate = 10
equilibration = 500
Niter = 2000

ins = Plots(
    temp=temp,
    box_scale=box_scale,
    thermo_rate=thermo_rate,
    equilibration=equilibration,
    Niter=Niter,
)

# ins.load_pos_config()
# x_vals, running_sum = ins.combine_cdf()
# fig, ax = plt.subplots()
# ax.plot(x_vals / ins.sigma, running_sum)
# ax.set_xlim(0, ins.L * np.sqrt(3) / (2 * ins.sigma))
# ax.axvline(ins.Rcut / ins.sigma, 0, ins.N)
# %%
fig, ax = plt.subplots()
ins.plot_cdf(label=f"{temp} K", fig=fig, ax=ax, linestyle="--")

# %%
