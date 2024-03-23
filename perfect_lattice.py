# %%
from RDF import RDF
from Plots import Plots
import matplotlib.pyplot as plt

file_format = "png"
dpi = 400

temp = 94
thermo_rate = 10
box_scale = 1
rcut_scale = 1
equilibration = 500
Niter = 2000

# %%
"""
 RDF PLOT 
"""

ins = Plots(
    temp=temp,
    box_scale=box_scale,
    rcut_scale=rcut_scale,
    thermo_rate=thermo_rate,
    equilibration=equilibration,
    Niter=Niter,
)

# %%
fig, ax = plt.subplots()
ins.plot_specific_rdf(time_snapshot=0, label=r"perfect lattice", fig=fig, ax=ax)
ax.set_title(
    rf"T={temp} K, L={box_scale}, R$_{{cut}}$={rcut_scale}",
    fontsize=20,
    fontweight="bold",
)
# %%
"""
CDF Plot
"""
fig, ax = plt.subplots()
ins.plot_specific_cdf(time_snapshot=0, label=r"perfect lattice", fig=fig, ax=ax)

# %%
