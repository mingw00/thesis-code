from networks_sparse import UNet as UNet_s
import torch
from partial_radon import PartialRadon
from data_management import IPDataset, SimulateMeasurements
import numpy as np
import config  
import torch.nn.functional as F
from networks import UNet
import torch.nn as nn

net_params = {
    "n_channels": 1,
    "n_classes": 1,
}


device = torch.device("cuda:0")

'''
Model B
'''

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

cnet_b = ConcatenatedUNet5()
cnet_b.load_state_dict(torch.load('results/model_B_geo_log_run_1.pt', map_location=device))
cnet_b.eval()

'''
Model D
'''

class ConcatenatedUNet(nn.Module):
    def __init__(self):
        super(ConcatenatedUNet, self).__init__()
        self.net4 = UNet(**net_params).to(device)
        self.net3 = UNet(**net_params).to(device)
        self.net2 = UNet(**net_params).to(device)
        self.net1 = UNet(**net_params).to(device)
        self.net0 = UNet(**net_params).to(device)


    def forward(self, x):
        x = self.net4(x)
        x = self.net3(x)
        x = self.net2(x)
        x = self.net1(x)
        x = self.net0(x)
        return x


cnet_d = ConcatenatedUNet()
cnet_d.load_state_dict(torch.load('results/Model_D_run_1.pt', map_location=device))
cnet_d.eval()

'''
Model A
'''
class ConcatenatedUNet(nn.Module):
    def __init__(self):
        super(ConcatenatedUNet, self).__init__()
        self.net4 = UNet(**net_params).to(device)
        self.net3 = UNet(**net_params).to(device)
        self.net2 = UNet(**net_params).to(device)
        self.net1 = UNet(**net_params).to(device)
        self.net0 = UNet(**net_params).to(device)

        # Load the weights for net0
        self.net4.load_state_dict(torch.load('results/model_A_geo_step_0.pt', map_location=device))
        self.net3.load_state_dict(torch.load('results/model_A_geo_step_1.pt', map_location=device))
        self.net2.load_state_dict(torch.load('results/model_A_geo_step_2.pt', map_location=device))
        self.net1.load_state_dict(torch.load('results/model_A_geo_step_3.pt', map_location=device))
        self.net0.load_state_dict(torch.load('results/model_A_geo_step_4.pt', map_location=device))
        self.net4.eval()
        self.net3.eval()
        self.net2.eval()
        self.net1.eval()
        self.net0.eval()
    def forward(self, x):
        x = self.net4(x)
        x = self.net3(x)
        x = self.net2(x)
        x = self.net1(x)
        x = self.net0(x)
        return x
cnet_a = ConcatenatedUNet()
cnet_a.eval()
'''
Model C
'''
class ConcatenatedUNet(nn.Module):
    def __init__(self):
        super(ConcatenatedUNet, self).__init__()
        self.net4 = UNet(**net_params).to(device)
        self.net3 = UNet(**net_params).to(device)
        self.net2 = UNet(**net_params).to(device)
        self.net1 = UNet(**net_params).to(device)
        self.net0 = UNet(**net_params).to(device)

        # Load the weights for net0
        self.net4.load_state_dict(torch.load('results/model_C_arth_step_0.pt', map_location=device))
        self.net3.load_state_dict(torch.load('results/model_C_arth_step_1.pt', map_location=device))
        self.net2.load_state_dict(torch.load('results/model_C_arth_step_2.pt', map_location=device))
        self.net1.load_state_dict(torch.load('results/model_C_arth_step_3.pt', map_location=device))
        self.net0.load_state_dict(torch.load('results/model_C_arth_step_4.pt', map_location=device))
        self.net4.eval()
        self.net3.eval()
        self.net2.eval()
        self.net1.eval()
        self.net0.eval()
    def forward(self, x):
        x = self.net4(x)
        x = self.net3(x)
        x = self.net2(x)
        x = self.net1(x)
        x = self.net0(x)
        return x
cnet_c = ConcatenatedUNet()
cnet_c.eval()



theta = torch.tensor(np.linspace(0,180,180,endpoint=False))
n_steps = 1
signal_supported_inside_circle = False
pRad = PartialRadon(theta=theta,steps=n_steps, circle=signal_supported_inside_circle)

test_data_params = {
    "path": config.DATA_PATH,
    "device": device,
}
test_data = IPDataset
test_loader_params =  {"shuffle": False, "num_workers": 0}
test_loader_params = dict(test_loader_params)
test_data = test_data("test", **test_data_params)
data_load_test = torch.utils.data.DataLoader(
            test_data, batch_size=16, **test_loader_params
        )




loss_2_a, loss_1_a = 0, 0
loss_2_b, loss_1_b = 0, 0
loss_2_c, loss_1_c = 0, 0
loss_2_d, loss_1_d = 0, 0
# l2 loss

mseloss = torch.nn.MSELoss(reduction="sum")

def loss_func_2(pred, tar):
    return mseloss(pred, tar) / tar.shape[0]


## l1 loss

l1_loss = torch.nn.L1Loss(reduction="sum")

def loss_func_1(pred, tar):
    return l1_loss(pred, tar) / tar.shape[0]

for j, batch in enumerate(data_load_test):

    img, _ = batch
    img = img.to(device)
    PY = pRad(img)
    inp = PY[:,-1].unsqueeze(1)
    tar = PY[:,0].unsqueeze(1)
    with torch.no_grad():
        pred_a = cnet_a(inp).detach()
        pred_b = cnet_b(inp).detach()
        pred_c = cnet_c(inp).detach()
        pred_d = cnet_d(inp).detach()
    
    loss_1_a += loss_func_1(pred_a, tar)
    loss_2_a += loss_func_2(pred_a, tar)
    loss_1_b += loss_func_1(pred_b, tar)
    loss_2_b += loss_func_2(pred_b, tar)   
    loss_1_c += loss_func_1(pred_c, tar)
    loss_2_c += loss_func_2(pred_c, tar)
    loss_1_d += loss_func_1(pred_d, tar)
    loss_2_d += loss_func_2(pred_d, tar)
    

loss_1_a /= j+1
loss_2_a /= j+1
loss_1_b /= j+1
loss_2_b /= j+1
loss_1_c /= j+1
loss_2_c /= j+1
loss_1_d /= j+1
loss_2_d /= j+1

print('Model A traied on geomoetric sinograms l1 loss = {:.2f}, l2 loss = {:.2f}'.format(loss_1_a,loss_2_a))
print('Model B traied on geomoetric sinograms l1 loss = {:.2f}, l2 loss = {:.2f}'.format(loss_1_b,loss_2_b))
print('Model C traied on geomoetric sinograms l1 loss = {:.2f}, l2 loss = {:.2f}'.format(loss_1_c,loss_2_c))
print('Model D traied on geomoetric sinograms l1 loss = {:.2f}, l2 loss = {:.2f}'.format(loss_1_d,loss_2_d))