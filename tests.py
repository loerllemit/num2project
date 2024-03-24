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

temp = 94
box_scale = 1
rcut_scale = 2
thermo_rate = 10
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
