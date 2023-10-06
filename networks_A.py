import os

from abc import ABCMeta, abstractmethod
from collections import OrderedDict


import pandas as pd
import torch

from tqdm import tqdm

from operators import l2_error
from partial_radon import PartialRadonStep

from unet_parts import *
import numpy as np

'''
adapted from https://github.com/jmaces/robust-nets/blob/master/ellipses/networks.py
'''

# ----- ----- Abstract Base Network ----- -----


class InvNet(torch.nn.Module, metaclass=ABCMeta):
    """ Abstract base class for networks solving linear inverse problems.

    The network is intended for the denoising of a direct inversion of a 2Dssss
    signal from (noisy) linear measurements. The measurement model

        y = Ax + noise

    can be used to obtain an approximate reconstruction x_ from y using, e.g.,
    the pseudo-inverse of A. The task of the network is either to directly
    obtain x from y or denoise and improve this first inversion x_ towards x.

    """

    def __init__(self):
        super(InvNet, self).__init__()

    @abstractmethod
    def forward(self, z):
        """ Applies the network to a batch of inputs z. """
        pass

    def freeze(self):
        """ Freeze all model weights, i.e. prohibit further updates. """
        for param in self.parameters():
            param.requires_grad = False

    @property
    def device(self):
        return next(self.parameters()).device

    def _train_step(
        self, batch, loss_func_2, optimizer, acc_steps, theta, n_steps, signal_supported_inside_circle, geometric, index
    ):
        '''
        k = index
        generate g^k, g^{k+1} on the fly
        '''
        inp_img, _ = batch
        inp_img = inp_img.to(self.device)
 

        pRadStep = PartialRadonStep(theta=theta,steps=n_steps,circle=signal_supported_inside_circle, geometric=geometric)
        PY = pRadStep(x = inp_img, index = index)
        inp = PY[:,-1].unsqueeze(1)
        tar = PY[:,0].unsqueeze(1)
        
        
        pred = self.forward(inp)
        ## l2 loss
        loss = loss_func_2(pred, tar)
        

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        

        ## with accumulation 
        '''
        if (batch_idx // batch_size + 1) % acc_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        return loss * acc_steps, inp, tar, pred
        '''
        return loss, inp, tar, pred

    def _val_step(self, batch, loss_func_2, theta, n_steps, signal_supported_inside_circle, geometric, index):
        inp_img, _ = batch
        inp_img = inp_img.to(self.device)
        
        pRadStep = PartialRadonStep(theta=theta,steps=n_steps,circle=signal_supported_inside_circle, geometric=geometric)
        PY = pRadStep(x = inp_img, index = index)
        inp = PY[:,-1].unsqueeze(1)
        tar = PY[:,0].unsqueeze(1)
        pred = self.forward(inp)

         ## l2 loss
        loss = loss_func_2(pred, tar)
                    
        return loss, inp, tar, pred

    def _on_epoch_end(
        self,
        epoch,
        save_epochs,
        save_path,
        logging,
        loss,
        tar,
        pred,
        v_loss,
        v_tar,
        v_pred,
    ):

        self._print_info()
        '''
        logging = logging.append(
            {
                "loss": loss.item(),
                "val_loss": v_loss.item(),
                "rel_l2_error": l2_error(
                    pred, tar, relative=True, squared=True
                )[0].item(),
                "val_rel_l2_error": l2_error(
                    v_pred, v_tar, relative=True, squared=True
                )[0].item(),
            },
            ignore_index=True,  
            sort=False,
        )
        '''
        
        
        logging = pd.concat(
            [logging,
            pd.DataFrame.from_dict({
                "loss": [loss.item()],
                "val_loss": [v_loss.item()],
                "rel_l2_error": [l2_error(
                    pred, tar, relative=True, squared=False
                )[0].item()],
                "val_rel_l2_error": [l2_error(
                    v_pred, v_tar, relative=True, squared=False
                )[0].item()],
            })],
            ignore_index=True,  
            sort=False,
        )
        
        print(logging.tail(1))

        if (epoch + 1) % save_epochs == 0:
            
            os.makedirs(save_path, exist_ok=True)
       
            torch.save(
                self.state_dict(),
                os.path.join(
                    save_path, "model_weights_epoch{}.pt".format(epoch + 1)
                ),
            )
        
            logging.to_pickle(
                os.path.join(
                    save_path, "losses_epoch{}.pkl".format(epoch + 1)
                ),
            )
            

        return logging


    def _add_to_progress_bar(self, dict):
        """ Can be overwritten by child classes to add to progress bar. """
        return dict

    def _on_train_end(self, save_path, logging):
        os.makedirs(save_path, exist_ok=True)
        torch.save(
            self.state_dict(), os.path.join(save_path, "model_weights.pt")
        )
        logging.to_pickle(os.path.join(save_path, "losses.pkl"),)

    def _print_info(self):
        """ Can be overwritten by child classes to print at epoch end. """
        pass

    def train_on(
        self,
        train_data,
        val_data,
        num_epochs,
        batch_size,
        loss_func_2,
        save_path,
        theta,
        n_steps,
        save_epochs=50,
        optimizer=torch.optim.Adam,
        optimizer_params={"lr": 2e-4, "eps": 1e-5},
        scheduler=torch.optim.lr_scheduler.StepLR,
        scheduler_params={"step_size": 1, "gamma": 1.0},
        acc_steps=1,
        train_transform=None,
        val_transform=None,
        train_loader_params={"shuffle": True},
        val_loader_params={"shuffle": False},
        signal_supported_inside_circle=False,
        geometric = False,
        index = 4
    ):
        optimizer = optimizer(self.parameters(), **optimizer_params)
        scheduler = scheduler(optimizer, **scheduler_params)

        train_data.transform = train_transform
        val_data.transform = val_transform

        train_loader_params = dict(train_loader_params)
        val_loader_params = dict(val_loader_params)
        if "sampler" in train_loader_params:
            train_loader_params["sampler"] = train_loader_params["sampler"](
                train_data
            )
        if "sampler" in val_loader_params:
            val_loader_params["sampler"] = val_loader_params["sampler"](
                val_data
            )

        data_load_train = torch.utils.data.DataLoader(
            train_data, batch_size, **train_loader_params
        )
        data_load_val = torch.utils.data.DataLoader(
            val_data, batch_size, **val_loader_params
        )

        logging = pd.DataFrame(
            columns=["loss", "val_loss", "rel_l2_error", "val_rel_l2_error"]
        )



        for epoch in range(num_epochs):
            self.train()  # make sure we are in train mode
            t = tqdm(
                enumerate(data_load_train),
                desc="epoch {} / {}".format(epoch, num_epochs),
                total=-(-len(train_data) // batch_size),
            )
            optimizer.zero_grad()
            loss = 0.0
            for i, batch in t:
                loss_b, inp, tar, pred = self._train_step(
                    batch, loss_func_2, optimizer, acc_steps, theta, n_steps, signal_supported_inside_circle, geometric, index
                )
                t.set_postfix(
                    **self._add_to_progress_bar({"loss": loss_b.item()})
                )
                with torch.no_grad():
                    loss += loss_b
            loss /= i + 1

            with torch.no_grad():
                self.eval()  # make sure we are in eval mode
                scheduler.step()
                v_loss = 0.0
                for i, v_batch in enumerate(data_load_val):
                    v_loss_b, v_inp, v_tar, v_pred = self._val_step(
                        v_batch, loss_func_2, theta, n_steps, signal_supported_inside_circle, geometric, index
                    )
                    
                    v_loss += v_loss_b
                v_loss /= i + 1

                logging = self._on_epoch_end(
                    epoch,
                    save_epochs,
                    save_path,
                    logging,
                    loss,
                    tar,
                    pred,
                    v_loss,
                    v_tar,
                    v_pred,
                )

        self._on_train_end(save_path, logging)
        return logging


# ----- ----- U-Net ----- -----
""" Full assembly of the parts to form the complete network """
class UNet(InvNet):
    def __init__(self, n_channels, n_classes, base_features=32, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, base_features))
        self.down1 = (Down(base_features, 2*base_features))
        self.down2 = (Down(2*base_features, 4*base_features))
        self.down3 = (Down(4*base_features, 8*base_features))
        factor = 2 if bilinear else 1
        self.down4 = (Down(8*base_features, 16*base_features // factor))
        self.up1 = (Up(16*base_features, 8*base_features // factor, bilinear))
        self.up2 = (Up(8*base_features, 4*base_features // factor, bilinear))
        self.up3 = (Up(4*base_features, 2*base_features // factor, bilinear))
        self.up4 = (Up(2*base_features, base_features, bilinear))
        self.outc = (OutConv(base_features, n_classes))
       

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        ##add sigmoid
        #logits_s = self.sigmoid(logits)
        return logits

    
