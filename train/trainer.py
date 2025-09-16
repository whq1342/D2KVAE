from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
import copy
from utils.scheduler import CosineAnnealingWarmupRestarts

class Trainer():
    def __init__(self, *, dataloader_dict: dict, model, epoch: int, lr: float, scheduler=None, verbose=True,
                 betaSchedule=None, device, zscore, args):
        self.dataloader_dict = dataloader_dict
        self.model = model
        self.lr = lr
        self.epoch = epoch
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = scheduler
        self.zscore = zscore
        self.args = args
        if self.scheduler is not None:
            print('Using CosineAnnealingWarmupRestarts lr.')
            self.scheduler = CosineAnnealingWarmupRestarts(self.optimizer,
                                                           first_cycle_steps=300,
                                                           cycle_mult=1.0,
                                                           max_lr=self.lr,
                                                           min_lr=0.0001,
                                                           warmup_steps=60,
                                                           gamma=0.5)
        self.betaSchedule = betaSchedule
        if self.betaSchedule is not None:
            print('Using monotonic beta schedule, cycle:{}, ratio:{}'.format(*self.betaSchedule))
            self.betaSchedule = self.frange_cycle_linear(0., 1., self.epoch, self.betaSchedule[0], self.betaSchedule[1])
        self.verbose = verbose
        print(f"Using {self.device} device")
        print("PyTorch Version: ", torch.__version__)
        self.model.to(self.device)

    def frange_cycle_linear(self, start, stop, n_epoch, n_cycle=4, ratio=0.5):
        L = np.ones(n_epoch)
        period = n_epoch / n_cycle
        step = (stop - start) / (period * ratio)  # linear schedule
        for c in range(n_cycle):
            v, i = start, 0
            while v <= stop and (int(i + c * period) < n_epoch):
                L[int(i + c * period)] = v
                v += step
                i += 1
        return L

    def loss_func(self, inputs_x, labels, index):
        inputs_x = inputs_x.transpose(0, 1)
        labels = labels.reshape(-1)[torch.nonzero(labels.reshape(-1))]
        
        self.z_x, self.z_x_mean, self.z_x_logvar, self.z_x_mean_p, self.z_x_logvar_p = self.model.inference(inputs_x)
        x_c = self.model.generation_x(self.z_x)
        
        y_mean, y_logvar, h_koopman, h_real, z_y_mean, z_y_logvar, z_y_mean_p, z_y_logvar_p = self.model.regression(self.z_x, index)

        reconstruct_loss = F.mse_loss(inputs_x, x_c)
        kld_x_loss = -0.5 * torch.mean(self.z_x_logvar - self.z_x_logvar_p - torch.div((self.z_x_logvar.exp() + (self.z_x_mean - self.z_x_mean_p).pow(2)), self.z_x_logvar_p.exp()))
        kld_y_loss = -0.5 * torch.mean(z_y_logvar - z_y_logvar_p - torch.div((z_y_logvar.exp() + (z_y_mean - z_y_mean_p).pow(2)), z_y_logvar_p.exp()))

        label_loss = F.mse_loss(labels, y_mean)
        koopman_loss = F.mse_loss(h_real, h_koopman)

        if self.betaSchedule is not None:
            kld_x_loss = kld_x_loss * self.betaSchedule[self._cur_epoch]
            kld_y_loss = kld_y_loss * self.betaSchedule[self._cur_epoch]
            
        loss = reconstruct_loss + self.args.label_weight * label_loss + self.args.kl_x_weight * kld_x_loss + self.args.kl_y_weight * kld_y_loss + self.args.koopman_weight * koopman_loss
        return loss, reconstruct_loss, kld_x_loss, label_loss, kld_y_loss, koopman_loss


    def _do_epoch(self, phase: str):
        avg_loss = 0.
        avg_reconstruct_loss = 0.
        avg_kld_x_loss = 0.
        avg_label_loss = 0.
        avg_kld_y_loss = 0.
        avg_koopman_loss = 0.
        sample_num = len(self.dataloader_dict[phase].sampler)
        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()
        for inputs, labels, index in self.dataloader_dict[phase]:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            index = index.to(self.device)
            self.optimizer.zero_grad()
            loss, reconstruct_loss, kld_x_loss, label_loss, kld_y_loss, koopman_loss = self.loss_func(inputs,labels,index)

            if phase == 'train':
                loss.backward()
                self.optimizer.step()
            avg_loss += loss.item() * inputs.size(0)
            avg_reconstruct_loss += reconstruct_loss.item() * inputs.size(0)
            avg_kld_x_loss += kld_x_loss.item() * inputs.size(0)
            avg_label_loss += label_loss.item() * inputs.size(0)
            avg_kld_y_loss += kld_y_loss.item() * inputs.size(0)
            avg_koopman_loss += koopman_loss.item() * inputs.size(0)

        return [avg_loss / sample_num, avg_reconstruct_loss / sample_num,
                avg_kld_x_loss / sample_num, avg_label_loss / sample_num,
                avg_kld_y_loss / sample_num, avg_koopman_loss / sample_num]

    def train(self):
        best_layer_wts = copy.deepcopy(self.model.state_dict())
        best_loss = np.inf

        for self._cur_epoch in range(self.epoch):
            # start_time = time.time()
            train_loss = self._do_epoch("train")
            if self.scheduler is not None:
                self.scheduler.step()
                if self._cur_epoch % 10 == 0 and self.verbose:
                    print('last lr: ', self.scheduler.get_lr())
            with torch.no_grad():
                val_loss = self._do_epoch("val")
            if val_loss[3] < best_loss:
                best_loss = val_loss[3]
                best_label = val_loss[3]
                best_layer_wts = copy.deepcopy(self.model.state_dict())
                best_epoch = self._cur_epoch
                torch.save(self.model.state_dict(), f"./models/label_rate_{self.args.label_rate}.pth")
            if self._cur_epoch % 10 == 0 and self.verbose:
                print(
                    f"Epoch {self._cur_epoch + 1}/{self.epoch}, \ntrain total: {train_loss[0]:>5f}, reconstruct: {train_loss[1]:>5f}, kld_x: {train_loss[2]:>5f}, label: {train_loss[3]:>5f}, kld_y: {train_loss[4]:>5f}, koopman: {train_loss[5]:>5f} \nval total: {val_loss[0]:>5f}, reconstruct: {val_loss[1]:>5f}, kld_x: {val_loss[2]:>5f}, label: {val_loss[3]:>5f}, kld_y: {val_loss[4]:>5f}, koopman: {val_loss[5]:>5f}")
        self.model.load_state_dict(best_layer_wts)
        print(f'Best loss: {best_loss:>7f}, best label loss: {best_label:>7f}, @ epoch: {best_epoch}')
        return best_epoch