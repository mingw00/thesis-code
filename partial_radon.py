import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from pytorch_radon.utils import PI, SQRT2, deg2rad, affine_grid, grid_sample

# TO DO:
# * make efficient
# * investigate any mismatch between pytorch_radon.Radon and final timestep of PartialRadon
# * fix tilting in inverse transform (may originate in Radon)
# * allow uneven timesteps
# * check all arguments: 
#       - assert theta in [0,180)

def rad2deg(r, dtype=torch.float):
    return (r*180/PI).to(dtype)

def hard_mask(t,wid,normalized=False):
    ran = torch.arange(wid) - (wid-1)/2
    mask = (ran.abs() < t).repeat([1,1,wid,1]).transpose(-1,-2)
    if normalized:
        return mask/(1.*mask.sum(-2))
    return mask

class PartialRadon(nn.Module):
    def __init__(self, in_size=None, theta=None, steps=None, circle=True, dtype=torch.float, geometric=False):
        super(PartialRadon, self).__init__()
        self.circle = circle
        self.theta = theta
        if theta is None:
            self.theta = torch.arange(180)
        self.steps = steps
        if steps is None:
            self.steps = 1
        self.dtype = dtype
        self.all_grids = None
        self.geometric = geometric
        if in_size is not None:
            self.all_grids = self._create_grids(self.theta, in_size, circle)

    def forward(self, x):
        N, C, W, H = x.shape
        assert C == 1 # this channel will be replaced by time steps.
        assert W == H

        if self.all_grids is None:
            self.all_grids = self._create_grids(self.theta, W, self.circle)

        if not self.circle:
            diagonal = SQRT2 * W
            pad = int((diagonal - W).ceil())
            new_center = (W + pad) // 2
            old_center = W // 2
            pad_before = new_center - old_center
            pad_width = (pad_before, pad - pad_before)
            x = F.pad(x, (pad_width[0], pad_width[1], pad_width[0], pad_width[1]))

        N, C, W, _ = x.shape
        out = torch.zeros(N, 1+self.steps, W, len(self.theta), device=x.device, dtype=self.dtype)
        if self.geometric:
            segments = np.logspace(np.log10(1),np.log10(W/2), num=self.steps+1, dtype=int)
        else:
            segments = np.linspace(1,W/2,self.steps+1,dtype=int)

     

        for i in range(len(self.theta)):
            rotated = grid_sample(x, self.all_grids[i].repeat(N, 1, 1, 1).to(x.device))
            
            for step in range(1+self.steps):
                mask_step = hard_mask(segments[step],W,normalized=True).to(x.device)
                out[:,step,:,i] = (rotated*mask_step).sum(-2)[:,0] # TO DO: Make work for batch ([:,0])

        return out

    def _create_grids(self, angles, grid_size, circle):
        if not circle:
            grid_size = int((SQRT2*grid_size).ceil())
        grid_shape = [1, 1, grid_size, grid_size]
        all_grids = []
        for theta in angles:
            theta = deg2rad(theta, self.dtype)
            R = torch.tensor([[
                [theta.cos(), theta.sin(), 0],
                [-theta.sin(), theta.cos(), 0],
            ]], dtype=self.dtype)
            all_grids.append(affine_grid(R, grid_shape))
        return all_grids


class IPartialRadon(nn.Module):
    def __init__(self, in_size=None, theta=None, steps=None, circle=True, dtype=torch.float):
        # steps is a dummy variable, not used
        super(IPartialRadon, self).__init__()
        self.circle = circle
        self.theta = theta if theta is not None else torch.arange(180)
        self.in_size = in_size
        self.dtype = dtype
        self.rad2deg = lambda x: rad2deg(x, dtype)
        self.ygrid, self.xgrid, self.all_grids = None, None, None
        if in_size is not None:
            self.ygrid, self.xgrid = self._create_yxgrid(in_size, circle)
            self.full_grid = self._create_fullgrid(in_size, circle)

    def forward(self, x):
        # x has shape N, S, W, A, where
        # N is batch size,
        # S is the number of "time steps",
        # W is the line resolution,
        # A is the angle resolution.
        # Only x[:,0] is used for inversion.

        # Side length of square containing the disk
        # on which the signal will be supported.
        # If not self.circle, then it_size > in_size.
        it_size = x.shape[2]
        ch_size = 1 # output should only have one channel
        ba_size = x.shape[0]

        if self.in_size is None:
            self.in_size = int((it_size/SQRT2).floor()) if not self.circle else it_size
        if None in [self.ygrid, self.xgrid, self.all_grids]:
            self.ygrid, self.xgrid = self._create_yxgrid(self.in_size, self.circle)
            self.full_grid = self._create_fullgrid(self.in_size, self.circle)

        reco = grid_sample(x[:,0:1],self.full_grid.repeat(ba_size,1,1,1).to(x.device))

        if not self.circle:
            W = self.in_size
            diagonal = it_size
            pad = int(torch.tensor(diagonal - W, dtype=torch.float).ceil())
            new_center = (W + pad) // 2
            old_center = W // 2
            pad_before = new_center - old_center
            pad_width = (pad_before, pad - pad_before)
            reco = F.pad(reco, (-pad_width[0], -pad_width[1], -pad_width[0], -pad_width[1]))

        if self.circle:
            reconstruction_circle = (self.xgrid ** 2 + self.ygrid ** 2) <= 1
            reconstruction_circle = reconstruction_circle.repeat(ba_size, ch_size, 1, 1)
            reco[~reconstruction_circle] = 0.

        return reco

    def _create_yxgrid(self, in_size, circle):
        if not circle:
            in_size = int((SQRT2*in_size).ceil())
        unitrange = torch.linspace(-1, 1, in_size, dtype=self.dtype)
        return torch.meshgrid(unitrange, unitrange)

    def _create_fullgrid(self, in_size, circle):
        # TO DO: Fix tilt, clean up, and make efficient
        if not circle:
            in_size = int((SQRT2*in_size).ceil())
        thetagrid = self.rad2deg(torch.atan2(self.ygrid,self.xgrid))
        thetagrid[ thetagrid >= 0 ] = 180. - thetagrid[ thetagrid >= 0 ]
        thetagrid = thetagrid.abs()
        thetagrid = (thetagrid - 90.)/90. # grid_sample needs values in [-1,1]
        rgrid = (self.xgrid**2 + self.ygrid**2).sqrt()*(1.-2.*(self.ygrid>=0))
        return torch.stack((thetagrid,rgrid),dim=-1)



class PartialRadonStep(nn.Module):
    def __init__(self, in_size=None, theta=None, steps=None, circle=True, dtype=torch.float, geometric=False):
        super(PartialRadonStep, self).__init__()
        self.circle = circle
        self.theta = theta
        if theta is None:
            self.theta = torch.arange(180)
        self.steps = steps
        if steps is None:
            self.steps = 1
        self.dtype = dtype
        self.all_grids = None
        self.geometric = geometric
        if in_size is not None:
            self.all_grids = self._create_grids(self.theta, in_size, circle)

    def forward(self, x, index):
        N, C, W, H = x.shape
        assert C == 1 # this channel will be replaced by time steps.
        assert W == H

        if self.all_grids is None:
            self.all_grids = self._create_grids(self.theta, W, self.circle)

        if not self.circle:
            diagonal = SQRT2 * W
            pad = int((diagonal - W).ceil())
            new_center = (W + pad) // 2
            old_center = W // 2
            pad_before = new_center - old_center
            pad_width = (pad_before, pad - pad_before)
            x = F.pad(x, (pad_width[0], pad_width[1], pad_width[0], pad_width[1]))

        N, C, W, _ = x.shape
        out = torch.zeros(N, 2, W, len(self.theta), device=x.device, dtype=self.dtype)

        if self.geometric:
            segments = np.logspace(np.log10(1),np.log10(W/2), num=self.steps+1, dtype=int)
        else:
            segments = np.linspace(1,W/2,self.steps+1,dtype=int)


        for i in range(len(self.theta)):
            rotated = grid_sample(x, self.all_grids[i].repeat(N, 1, 1, 1).to(x.device))
            
            for step in range(2):
                mask_step = hard_mask(segments[4 - index + step],W,normalized=True).to(x.device)
                out[:,step,:,i] = (rotated*mask_step).sum(-2)[:,0] # TO DO: Make work for batch ([:,0])

        return out

    def _create_grids(self, angles, grid_size, circle):
        if not circle:
            grid_size = int((SQRT2*grid_size).ceil())
        grid_shape = [1, 1, grid_size, grid_size]
        all_grids = []
        for theta in angles:
            theta = deg2rad(theta, self.dtype)
            R = torch.tensor([[
                [theta.cos(), theta.sin(), 0],
                [-theta.sin(), theta.cos(), 0],
            ]], dtype=self.dtype)
            all_grids.append(affine_grid(R, grid_shape))
        return all_grids


class PartialRadonStepEnd(nn.Module):
    def __init__(self, in_size=None, theta=None, steps=None, circle=True, dtype=torch.float, geometric=False):
        super(PartialRadonStepEnd, self).__init__()
        self.circle = circle
        self.theta = theta
        if theta is None:
            self.theta = torch.arange(180)
        self.steps = steps
        if steps is None:
            self.steps = 1
        self.dtype = dtype
        self.all_grids = None
        self.geometric = geometric
        if in_size is not None:
            self.all_grids = self._create_grids(self.theta, in_size, circle)

    def forward(self, x, index):
        N, C, W, H = x.shape
        assert C == 1 # this channel will be replaced by time steps.
        assert W == H

        if self.all_grids is None:
            self.all_grids = self._create_grids(self.theta, W, self.circle)

        if not self.circle:
            diagonal = SQRT2 * W
            pad = int((diagonal - W).ceil())
            new_center = (W + pad) // 2
            old_center = W // 2
            pad_before = new_center - old_center
            pad_width = (pad_before, pad - pad_before)
            x = F.pad(x, (pad_width[0], pad_width[1], pad_width[0], pad_width[1]))

        N, C, W, _ = x.shape
        out = torch.zeros(N, 2, W, len(self.theta), device=x.device, dtype=self.dtype)

        if self.geometric:
            segments = np.logspace(np.log10(1),np.log10(W/2), num=self.steps+1, dtype=int)
        else:
            segments = np.linspace(1,W/2,self.steps+1,dtype=int)

        for i in range(len(self.theta)):
            rotated = grid_sample(x, self.all_grids[i].repeat(N, 1, 1, 1).to(x.device))
            mask_step = hard_mask(segments[4 - index],W,normalized=True).to(x.device)
            out[:,0,:,i] = (rotated*mask_step).sum(-2)[:,0] 
            mask_step = hard_mask(segments[-1],W,normalized=True).to(x.device)
            out[:,1,:,i] = (rotated*mask_step).sum(-2)[:,0]
           
        return out

    def _create_grids(self, angles, grid_size, circle):
        if not circle:
            grid_size = int((SQRT2*grid_size).ceil())
        grid_shape = [1, 1, grid_size, grid_size]
        all_grids = []
        for theta in angles:
            theta = deg2rad(theta, self.dtype)
            R = torch.tensor([[
                [theta.cos(), theta.sin(), 0],
                [-theta.sin(), theta.cos(), 0],
            ]], dtype=self.dtype)
            all_grids.append(affine_grid(R, grid_shape))
        return all_grids
    


class Radon(nn.Module):
    def __init__(self, in_size=None, theta=None, circle=True, dtype=torch.float):
        super(Radon, self).__init__()
        self.circle = circle
        self.theta = theta
        if theta is None:
            self.theta = torch.arange(180)
        self.dtype = dtype
        self.all_grids = None
        if in_size is not None:
            self.all_grids = self._create_grids(self.theta, in_size, circle)

    def forward(self, x):
        N, C, W, H = x.shape
        assert C == 1 # this channel will be replaced by time steps.
        assert W == H

        if self.all_grids is None:
            self.all_grids = self._create_grids(self.theta, W, self.circle)

        if not self.circle:
            diagonal = SQRT2 * W
            pad = int((diagonal - W).ceil())
            new_center = (W + pad) // 2
            old_center = W // 2
            pad_before = new_center - old_center
            pad_width = (pad_before, pad - pad_before)
            x = F.pad(x, (pad_width[0], pad_width[1], pad_width[0], pad_width[1]))

        N, C, W, _ = x.shape
        out = torch.zeros(N, 1, W, len(self.theta), device=x.device, dtype=self.dtype)

        segments = W/2
        #segments = np.linspace(1,W/2,self.steps+1,dtype=int)

        for i in range(len(self.theta)):
            rotated = grid_sample(x, self.all_grids[i].repeat(N, 1, 1, 1).to(x.device))
            mask_step = hard_mask(segments,W,normalized=True).to(x.device)
            out[:,0,:,i] = (rotated*mask_step).sum(-2)[:,0]
        return out

    def _create_grids(self, angles, grid_size, circle):
        if not circle:
            grid_size = int((SQRT2*grid_size).ceil())
        grid_shape = [1, 1, grid_size, grid_size]
        all_grids = []
        for theta in angles:
            theta = deg2rad(theta, self.dtype)
            R = torch.tensor([[
                [theta.cos(), theta.sin(), 0],
                [-theta.sin(), theta.cos(), 0],
            ]], dtype=self.dtype)
            all_grids.append(affine_grid(R, grid_shape))
        return all_grids

