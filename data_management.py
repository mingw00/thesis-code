import glob
import os

import numpy as np
import torch

from tqdm import tqdm


'''
Code from https://github.com/jmaces/robust-nets/blob/master/ellipses/data_management.py
'''

# ----- Dataset creation, saving, and loading -----


def create_iterable_dataset(
    n, set_params, generator, gen_params,
):
    """ Creates training, validation, and test data sets.

    Samples data signals from a data generator and stores them.

    Parameters
    ----------
    n : int
        Dimension of signals x.
    set_params : dictionary
        Must contain values for the following keys:
        path : str
            Directory path for storing the data sets.
        num_train : int
            Number of samples in the training set.
        num_val : int
            Number of samples in the validation set.
        num_test : int
            Number of samples in the validation set.
    generator : callable
        Generator function to create signal samples x. Will be called with
        the signature generator(n, **gen_params).
    gen_params : dictionary
        Additional keyword arguments passed on to the signal generator.
    """
    N_train, N_val, N_test = [
        set_params[key] for key in ["num_train", "num_val", "num_test"]
    ]

    def _get_signal():
        x, _ = generator(n, **gen_params)
        return x

    os.makedirs(os.path.join(set_params["path"], "train"), exist_ok=True)
    os.makedirs(os.path.join(set_params["path"], "val"), exist_ok=True)
    os.makedirs(os.path.join(set_params["path"], "test"), exist_ok=True)

    for idx in tqdm(range(N_train), desc="generating training signals"):
        torch.save(
            _get_signal(),
            os.path.join(
                set_params["path"], "train", "sample_{}.pt".format(idx)
            ),
        )

    for idx in tqdm(range(N_val), desc="generating validation signals"):
        torch.save(
            _get_signal(),
            os.path.join(
                set_params["path"], "val", "sample_{}.pt".format(idx)
            ),
        )

    for idx in tqdm(range(N_test), desc="generating test signals"):
        torch.save(
            _get_signal(),
            os.path.join(
                set_params["path"], "test", "sample_{}.pt".format(idx)
            ),
        )



class IPDataset(torch.utils.data.Dataset):
    """ Datasets for imaging inverse problems.

    Loads image signals created by `create_iterable_dataset` from a directory.

    Implements the map-style dataset in `torch`.

    Attributed
    ----------
    subset : str
        One of "train", "val", "test".
    path : str
        The directory path. Should contain the subdirectories "train", "val",
        "test" containing the training, validation, and test data respectively.
    """

    def __init__(self, subset, path, transform=None, device=None):
        self.path = path
        self.files = glob.glob(os.path.join(path, subset, "*.pt"))
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # load image and add channel dimension
        if self.device is not None:
            out = (torch.load(self.files[idx]).unsqueeze(0).to(self.device),)
        else:
            out = (torch.load(self.files[idx]).unsqueeze(0),)
        #return self.transform(out) if self.transform is not None else out
        return self.transform(out) if self.transform is not None else (torch.load(self.files[idx]).unsqueeze(0).to(self.device),0)




class SimulateMeasurements(object):
    """ Forward operator on target samples.

    Computes measurements and returns (measurement, target) pair.

    Parameters
    ----------
    operator : callable
        The measurement operation to use.

    """

    def __init__(self, operator):
        self.operator = operator

    def __call__(self, target):
        (target,) = target
        meas = self.operator(target)
        return meas, target



class CenterCrop(object):
    """ Crops (input, target) image pairs to have matching size. """

    def __init__(self, shape):
        self.shape = shape

    def __call__(self, imgs):
        return tuple([transforms.center_crop(img, self.shape) for img in imgs])


class Flatten(object):
    """ Flattens selected dimensions of tensors. """

    def __init__(self, start_dim, end_dim):
        self.start_dim = start_dim
        self.end_dim = end_dim

    def __call__(self, inputs):
        return tuple(
            [torch.flatten(x, self.start_dim, self.end_dim) for x in inputs]
        )


class Normalize(object):
    """ Normalizes (input, target) pairs with respect to target or input. """

    def __init__(self, p=2, reduction="sum", use_target=True):
        self.p = p
        self.reduction = reduction
        self.use_target = use_target

    def __call__(self, inputs):
        inp, tar = inputs
        norm = torch.norm(tar if self.use_target else inp, p=self.p)
        if self.reduction == "mean" and not self.p == "inf":
            norm /= np.prod(tar.shape) ** (1 / self.p)
        return inputs[0] / norm, inputs[1] / norm



# ---- run data generation -----
if __name__ == "__main__":
    import config
    print(torch.cuda.is_available())
    np.random.seed(config.numpy_seed)
    torch.manual_seed(config.torch_seed)
    create_iterable_dataset(
        config.n, config.set_params, config.data_gen, config.data_params,
    )
