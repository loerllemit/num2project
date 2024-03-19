import numpy as np

xyz = (
    np.mgrid[
        0 : 10 : 10 / 10,
        0 : 10 : 10 / 10,
        0 : 10 : 10 / 5,
    ]
    .reshape(3, -1)
    .T
)

spacing = 1.0 * ins.sigma
np.arange(ins.L / 2 - spacing * 4, ins.L / 2 + spacing * 4, spacing)

x = np.linspace(0, 5, 4, endpoint=False)
y = np.linspace(0, 1, 3)
xv, yv = np.meshgrid(x, y)

np.array(np.meshgrid(x, y)).reshape(2, -1).T


xyz = (
    np.mgrid[
        np.linspace(1, 9, 9),
        1:10:1,
        1:9:1,
    ]
    .reshape(3, -1)
    .T
)

# %%
