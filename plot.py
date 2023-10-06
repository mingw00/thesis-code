from networks import UNet
from networks_sparse import UNet as UNet_s
import torch
from partial_radon import PartialRadon, Radon, IPartialRadon
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from data_management import IPDataset, SimulateMeasurements
import numpy as np
import config  # isort:skip
import torch.nn as nn
import torch.nn.functional as F


net_params = {
    "n_channels": 1,
    "n_classes": 1,
}

device = torch.device("cuda:0")

class ConcatenatedUNet2(nn.Module):
    def __init__(self):
        super(ConcatenatedUNet2, self).__init__()
        self.net4 = UNet(**net_params).to(device)
        self.net3 = UNet(**net_params).to(device)

    def forward(self, x):
        x = self.net4(x)
        x = self.net3(x)
        return x
class ConcatenatedUNet3(nn.Module):
    def __init__(self):
        super(ConcatenatedUNet3, self).__init__()
        self.cnet = ConcatenatedUNet2()
        self.net2 = UNet(**net_params).to(device)


    def forward(self, x):
        x = self.cnet(x)
        x = self.net2(x)
        return x   

class ConcatenatedUNet4(nn.Module):
    def __init__(self):
        super(ConcatenatedUNet4, self).__init__()
        self.cnet = ConcatenatedUNet3()
        self.net1 = UNet(**net_params).to(device)

    def forward(self, x):
        x = self.cnet(x)
        x = self.net1(x)
        return x   
class ConcatenatedUNet5(nn.Module):
    def __init__(self):
        super(ConcatenatedUNet5, self).__init__()
        self.cnet = ConcatenatedUNet4()
        self.net0 = UNet(**net_params).to(device)

    def forward(self, x):
        x = self.cnet(x)
        x = self.net0(x)
        return x   
'''
load Model B
'''
cnet_b = ConcatenatedUNet5()
cnet_b.load_state_dict(torch.load('results/model_B_geo_run_1.pt', map_location=device))
cnet_b.eval()

'''
load sparse network
'''
net_s = UNet_s(**net_params).to(device)
net_s.load_state_dict(torch.load('results/model_sparse.pt', map_location=device))
net_s.eval()

n_steps = 5
theta = torch.tensor(np.linspace(0,180,180,endpoint=False))
theta_s = torch.tensor(np.linspace(0,180,60,endpoint=False))

signal_supported_inside_circle = False
pRad = PartialRadon(theta=theta,steps=n_steps,circle=signal_supported_inside_circle, geometric=True)
pRad_s = Radon(theta=theta_s,circle=signal_supported_inside_circle)

test_data_params = {
    "path": config.DATA_PATH,
    "device": device,
}
test_data = IPDataset
test_loader_params =  {"shuffle": False, "num_workers": 0}
test_loader_params = dict(test_loader_params)
test_data = test_data("test", **test_data_params)
data_load_test = torch.utils.data.DataLoader(
            test_data, batch_size=1, **test_loader_params
        )



for i, batch in enumerate(data_load_test):
    if i > 500:
        break
    img, _ = batch

wid = 128
unitrange = torch.linspace(-1, 1, wid)

img = img.to(device)


vmin = 0
vmax =1

PY_s = pRad_s(img)
PY = pRad(img)
inp_s = F.interpolate(PY_s, size=[PY_s.size()[2], 180], mode='nearest')
pred_s = net_s(inp_s) # reconstruction form sparse sinorgams
pred_s = pred_s.detach()
pred_s = cnet_b(pred_s).detach()

inp =  PY[:,-1].unsqueeze(1)
pred = cnet_b(inp) # reconstruction form normal sinograms
pred = pred.detach()
tar =  PY[:,0].unsqueeze(1)
pred_i = PY.clone()
pred_i[:, 0] = pred.squeeze()
for i in range(n_steps-2):
    cnet_b = cnet_b.cnet
    pred_i[:,i+1] = cnet_b(inp).detach().squeeze()
pred_i[:,-2] = cnet_b.net4(inp).detach().squeeze()



fig, ax = plt.subplots(pred_i.shape[1],2, figsize=(24, 16))
fig.subplots_adjust(hspace=0.5)
plt.subplots_adjust(wspace=0.07)
im=0
'''
1st col: ground truth geometric sinograms
2nd col: at step k, plot U_k\circ\cdots\circ U_0(x_500)
'''

for step in range(pred_i.shape[1]):

    ax[step,1].imshow(pred_i[im,step].cpu().transpose(0,1),origin="lower",aspect="auto",extent=[unitrange[0],unitrange[-1],theta[0],theta[-1]],vmin=vmin,vmax=vmax)
    ax[step,1].set_ylabel(r"$\theta$")
    ax[step,1].set_xlabel("s")
    ax[step,1].set_title(f"step{5-step}")
    ax[step,0].imshow(PY[im,step].cpu().transpose(0,1),origin="lower",aspect="auto",extent=[unitrange[0],unitrange[-1],theta[0],theta[-1]],vmin=vmin,vmax=vmax)
    ax[step,0].set_ylabel(r"$\theta$")
    ax[step,0].set_xlabel("s")
    ax[step,0].set_title(f"step{5-step}")


plt.show()





ipRad = IPartialRadon(theta=theta,steps=n_steps,circle=signal_supported_inside_circle)
inv_s = ipRad(pred_s) # apply domain transform to pred_s
inv = ipRad(pred)
fig, ax = plt.subplots(1,2, figsize=(8, 4))
fig.subplots_adjust(hspace=0.5)

vmin = img[0].min()
vmax = img[0].max()

ax[0].imshow( inv_s[0,0].cpu(),origin="lower",vmin=vmin,vmax=vmax)
ax[0].set_xlabel("x")
ax[0].set_ylabel("y")
ax[0].set_title('From sparse sinogram (60 angles)')
ax[1].imshow( inv[0,0].cpu(), origin="lower",vmin=vmin,vmax=vmax)
ax[1].set_xlabel("x")
ax[1].set_ylabel("y")
ax[1].set_title("From normal sinogram (180 angles)")
plt.show()
