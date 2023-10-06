from networks import UNet, InvNet
import torch
import os
import matplotlib as mpl

from data_management import IPDataset, SimulateMeasurements
import numpy as np
import torch.nn as nn
# ----- load configuration -----
import config  # isort:skip

# ----- global configuration -----

mpl.use("agg")
device = torch.device("cuda:0")
#device = torch.device('cpu')


# ----- network configuration -----

net_params = {
    "n_channels": 1,
    "n_classes": 1,
    "base_features": 32
}
# ----- training configuration -----
## l2 loss

mseloss = torch.nn.MSELoss(reduction="sum")

def loss_func_2(pred, tar):
    return mseloss(pred, tar) / pred.shape[0]


## l1 loss

l1_loss = torch.nn.L1Loss(reduction="sum")

def loss_func_1(pred, tar):
    return l1_loss(pred, tar) / pred.shape[0]

'''
set geometric = True to use geometric sinogram
set geometric = False to use arithmetic sinogram
index means that U_index\circ\cdots\circ U_0 will be trained
'''

train_phases = 1
train_params = {
    "num_epochs": [25],
    "batch_size": [16],
    "loss_func_2": loss_func_2,
    "save_path": [
        os.path.join(
            config.RESULTS_PATH,
            "B_step_0_"
            "train_phase_{}".format((i + 1) % (train_phases + 1)),
        )
        for i in range(train_phases + 1)
    ],
    "theta": [torch.tensor(np.linspace(0,180,180,endpoint=False))],
    "n_steps": [5],
    "save_epochs": 1,
    "optimizer": torch.optim.Adam,
    "optimizer_params": [
        {"lr": 2e-4, "eps": 1e-5, "weight_decay": 1e-4}],
    "scheduler": torch.optim.lr_scheduler.StepLR,
    "scheduler_params": {"step_size": 1, "gamma": 1.0},
    "acc_steps": [1],
    "train_loader_params": {"shuffle": True, "num_workers": 0},
    "val_loader_params": {"shuffle": False, "num_workers": 0},
    "signal_supported_inside_circle": False,
    "geometric": True,
    "index": [0]
}
 
# ----- data configuration -----

train_data_params = {
    "path": config.DATA_PATH,
    "device": device,
}
train_data = IPDataset

val_data_params = {
    "path": config.DATA_PATH,
    "device": device,
}
val_data = IPDataset

# ------ save hyperparameters -------

os.makedirs(train_params["save_path"][-1], exist_ok=True)
with open(
    os.path.join(train_params["save_path"][-1], "hyperparameters.txt"), "w"
) as file:
    for key, value in net_params.items():
        file.write(key + ": " + str(value) + "\n")
    for key, value in train_params.items():
        file.write(key + ": " + str(value) + "\n")
    for key, value in train_data_params.items():
        file.write(key + ": " + str(value) + "\n")
    for key, value in val_data_params.items():
        file.write(key + ": " + str(value) + "\n")
    file.write("train_phases" + ": " + str(train_phases) + "\n")

# ------ construct network and train -----

'''
note that netk actually means UNet_{4-k}
'''
class ConcatenatedUNet2(InvNet):
    def __init__(self):
        super(ConcatenatedUNet2, self).__init__()
        self.net4 = UNet(**net_params).to(device)
        self.net3 = UNet(**net_params).to(device)
        '''
        load weights of U_0 obtained from previous training
        '''
        #self.net4.load_state_dict(torch.load('D:/thesis/results/final_end2end_step0_log_train_phase_1/model_weights_epoch16.pt', map_location=device))

    def forward(self, x):
        x = self.net4(x)
        x = self.net3(x)
        return x
class ConcatenatedUNet3(InvNet):
    def __init__(self):
        super(ConcatenatedUNet3, self).__init__()
        self.cnet = ConcatenatedUNet2()
        #self.cnet.load_state_dict(torch.load('D:/thesis/results/final_end2end_step1_log_train_phase_1/model_weights_epoch13.pt', map_location=device))
        self.net2 = UNet(**net_params).to(device)


    def forward(self, x):
        x = self.cnet(x)
        x = self.net2(x)
        return x   

class ConcatenatedUNet4(InvNet):
    def __init__(self):
        super(ConcatenatedUNet4, self).__init__()
        self.cnet = ConcatenatedUNet3()
        #self.cnet.load_state_dict(torch.load('D:/thesis/results/final_end2end_step2_log_train_phase_1/model_weights_epoch11.pt', map_location=device))
        self.net1 = UNet(**net_params).to(device)

    def forward(self, x):
        x = self.cnet(x)
        x = self.net1(x)
        return x   

class ConcatenatedUNet5(InvNet):
    def __init__(self):
        super(ConcatenatedUNet5, self).__init__()
        self.cnet = ConcatenatedUNet4()
        #self.cnet.load_state_dict(torch.load('D:/thesis/results/final_end2end_step3_log_train_phase_1/model_weights_epoch10.pt', map_location=device))
        self.net0 = UNet(**net_params).to(device)

    def forward(self, x):
        x = self.cnet(x)
        x = self.net0(x)
        return x   

cnet = UNet(**net_params).to(device) # train U_0
#cnet = ConcatenatedUNet5() #train U_4\circ\cdots\circ U_0
train_data = train_data("train", **train_data_params)
val_data = val_data("val", **val_data_params)

for i in range(train_phases):
    train_params_cur = {}
    for key, value in train_params.items():
        train_params_cur[key] = (
            value[i] if isinstance(value, (tuple, list)) else value
        )

    print("Phase {}:".format(i + 1))
    for key, value in train_params_cur.items():
        print(key + ": " + str(value))

    cnet.train_on(train_data, val_data, **train_params_cur)
