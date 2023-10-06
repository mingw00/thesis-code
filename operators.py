import math

from abc import ABC, abstractmethod

import numpy as np
import pytorch_radon
import skimage.transform
import torch
import torch_cg

'''
from https://github.com/jmaces/robust-nets/blob/master/ellipses/operators.py
'''

# ----- Utilities -----



def l2_error(X, X_ref, relative=False, squared=False, use_magnitude=True):
    """ Compute average l2-error of an image over last three dimensions.

    Parameters
    ----------
    X : torch.Tensor
        The input tensor of shape [..., 1, W, H] for real images or
        [..., 2, W, H] for complex images.
    X_ref : torch.Tensor
        The reference tensor of same shape.
    relative : bool, optional
        Use relative error. (Default False)
    squared : bool, optional
        Use squared error. (Default False)
    use_magnitude : bool, optional
        Use complex magnitudes. (Default True)

    Returns
    -------
    err_av :
        The average error.
    err :
        Tensor with individual errors.

    """
    assert X_ref.ndim >= 3  # do not forget the channel dimension

    if X_ref.shape[-3] == 2 and use_magnitude:  # compare complex magnitudes
        X_flat = torch.flatten(torch.sqrt(X.pow(2).sum(-3)), -2, -1)
        X_ref_flat = torch.flatten(torch.sqrt(X_ref.pow(2).sum(-3)), -2, -1)
    else:
        X_flat = torch.flatten(X, -3, -1)
        X_ref_flat = torch.flatten(X_ref, -3, -1)

    if squared:
        err = (X_flat - X_ref_flat).norm(p=2, dim=-1) ** 2
    else:
        err = (X_flat - X_ref_flat).norm(p=2, dim=-1)

    if relative:
        if squared:
            err = err / (X_ref_flat.norm(p=2, dim=-1) ** 2)
        else:
            err = err / X_ref_flat.norm(p=2, dim=-1)

    if X_ref.ndim > 3:
        err_av = err.sum() / np.prod(X_ref.shape[:-3])
    else:
        err_av = err
    return err_av.squeeze(), err


