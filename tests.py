# %%
from MolDyn import MolDyn
from RDF import RDF
from Plots import Plots
import matplotlib.pyplot as plt
import numpy as np
import time

# %%
"""
RUN MOLECULAR DYNAMICS
"""

ins = MolDyn(
    temp=300,
    thermo_rate=10,
    box_scale=1,
    rcut_scale=1,
    equilibration=500,
    Niter=2000,
)
# time the execution
start = time.time()
ins.run_md()
end = time.time()
print("execution time: ", end - start)


# %%
# """
#  TEST CDF
# """
# temp = 300
# box_scale = 1
# thermo_rate = 10
# equilibration = 500
# Niter = 2000

# ins = Plots(
#     temp=temp,
#     box_scale=box_scale,
#     thermo_rate=thermo_rate,
#     equilibration=equilibration,
#     Niter=Niter,
# )

# %%
# fig, ax = plt.subplots()
# ins.plot_cdf(label=f"{temp} K", fig=fig, ax=ax, linestyle="--")

# %%
