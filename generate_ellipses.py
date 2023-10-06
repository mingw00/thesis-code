import glob
import os

import numpy as np
import odl
import torch
from data_management import create_iterable_dataset
'''
Code from https://github.com/jmaces/robust-nets/blob/master/ellipses/data_management.py
'''
def sample_ellipses(
    n,
    c_min=10,
    c_max=20,
    max_axis=0.5,
    min_axis=0.05,
    margin_offset=0.3,
    margin_offset_axis=0.9,
    grad_fac=1.0,
    bias_fac=1.0,
    bias_fac_min=0.0,
    normalize=True,
    n_seed=None,
    t_seed=None,
):
    """ Creates an image of random ellipses.

    Creates a piecewise linear signal of shape n (two-dimensional) with a
    random number of ellipses with zero boundaries.
    The signal is created as a functions in the box [-1,1] x [-1,1].
    The image is generated such that it cannot have negative values.

    Parameters
    ----------
    n : tuple
        Height x width
    c_min : int, optional
        Minimum number of ellipses. (Default 10)
    c_max : int, optional
         Maximum number of ellipses. (Default 20)
    max_axis : double, optional
        Maximum radius of the ellipses. (Default .5, in [0, 1))
    min_axis : double, optional
        Minimal radius of the ellipses. (Default .05, in [0, 1))
    margin_offset : double, optional
        Minimum distance of the center coordinates to the image boundary.
        (Default .3, in (0, 1])
    margin_offset_axis : double, optional
        Offset parameter so that the ellipses to not touch the image boundary.
        (Default .9, in [0, 1))
    grad_fac : double, optional
        Specifies the slope of the random linear piece that is created for each
        ellipse. Set it to 0.0 for constant pieces. (Default 1.0)
    bias_fac : double, optional
        Scalar factor that upscales the bias of the linear/constant pieces of
        each ellipse. Essentially specifies the weights of the ellipsoid
        regions. (Default 1.0)
    bias_fac_min : double, optional
        Lower bound on the bias for the weights. (Default 0.0)
    normalize : bool, optional
        Normalizes the image to the interval [0, 1] (Default True)
    n_seed : int, optional
        Seed for the numpy random number generator for drawing the jump
        positions. Set to `None` for not setting any seed and keeping the
        current state of the random number generator. (Default `None`)
    t_seed : int, optional
        Seed for the troch random number generator for drawing the jump
        heights. Set to `None` for not setting any seed and keeping the
        current state of the random number generator. (Default `None`)

    Returns
    -------
    torch.Tensor
        Will be of shape n (two-dimensional).
    """

    if n_seed is not None:
        np.random.seed(n_seed)
    if t_seed is not None:
        torch.manual_seed(t_seed)

    c = np.random.randint(c_min, c_max)

    cen_x = (1 - margin_offset) * 2 * (np.random.rand(c) - 1 / 2)
    cen_y = (1 - margin_offset) * 2 * (np.random.rand(c) - 1 / 2)
    cen_max = np.maximum(np.abs(cen_x), np.abs(cen_y))

    ax_1 = np.minimum(
        min_axis + (max_axis - min_axis) * np.random.rand(c),
        (1 - cen_max) * margin_offset_axis,
    )
    ax_2 = np.minimum(
        min_axis + (max_axis - min_axis) * np.random.rand(c),
        (1 - cen_max) * margin_offset_axis,
    )

    weights = np.ones(c)
    rot = np.pi / 2 * np.random.rand(c)

    p = np.stack([weights, ax_1, ax_2, cen_x, cen_y, rot]).transpose()
    space = odl.discr.discr_sequence_space(n)

    coord_x = np.linspace(-1.0, 1.0, n[0])
    coord_y = np.linspace(-1.0, 1.0, n[1])
    m_x, m_y = np.meshgrid(coord_x, coord_y)

    X = np.zeros(n)
    for e in range(p.shape[0]):
        E = -np.ones(n)
        while E.min() < 0:
            E = odl.phantom.geometric.ellipsoid_phantom(
                space, p[e : (e + 1), :]
            ).asarray()
            E = E * (
                grad_fac * np.random.randn(1) * m_x
                + grad_fac * np.random.randn(1) * m_y
                + bias_fac_min
                + (bias_fac - bias_fac_min) * np.random.rand(1)
            )
        X = X + E

    X = torch.tensor(X, dtype=torch.float)

    if normalize:
        X = X / X.max()

    return X, torch.tensor(c, dtype=torch.float)

# ---- run data generation -----
if __name__ == "__main__":
    import config
    data_gen = sample_ellipses  # data generator function
    print(torch.cuda.is_available())
    np.random.seed(config.numpy_seed)
    torch.manual_seed(config.torch_seed)
    create_iterable_dataset(
        config.n, config.set_params, data_gen, config.data_params,
    )