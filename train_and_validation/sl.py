"""# Imports and Stuff..."""

import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import TensorDataset
from torchsummary import summary

import models
from dataset_handler import cifar10, fmnist, mnist
from dataset_handler.trigger import get_bd_set, GenerateTrigger
from helper import EarlyStopping

import csv
import gc

import torch

torch.manual_seed(47)
import numpy as np

np.random.seed(47)


class SLTrainAndValidation:
    def __init__(self, dataloaders, models, loss_fns, optimizers, lr_schedulers, early_stopping):
        self.dataloaders = dataloaders
        self.models = models
        self.loss_fns = loss_fns
        self.optimizers = optimizers
        self.lr_schedulers = lr_schedulers
        self.early_stopping = early_stopping

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        for name, model in self.models.items():
            if name.lower() != 'client':
                model.to(self.device)
        for model in self.models['client']:
            model.to(self.device)

        self.final_client_state_dict = self.models['client'][-1].state_dict()

        self.dataset_sizes = {k: len(v.dataset) for k, v in self.dataloaders.items() if k.lower() != 'train'}
        self.dataset_sizes['train'] = [len(item.dataset) for item in self.dataloaders['train']]

        self.num_batches = {k: len(v) for k, v in self.dataloaders.items() if k.lower() != 'train'}
        self.num_batches['train'] = [len(item) for item in self.dataloaders['train']]

    def step1_train(self, train_phase, phase_ds_num, trigger_obj, trig_ds, smpl_prctg, bd_label):

        phase = train_phase

        for name, model in self.models.items():
            if name.lower() != 'client':
                model.train()
        # for model in self.models['client']:
        #     model.train()

        running_loss1 = 0.0
        running_corrects1 = 0.0
        running_loss2 = 0.0
        running_corrects2 = 0.0
        epoch_loss = {'clean': 0.0, 'bd': 0.0}
        epoch_corrects = {'clean': 0.0, 'bd': 0.0}

        inputs = {}
        labels = {}
        bd_labels = {}
        bd_inputs = {}

        '''Iterating over multiple dataloaders simultaneously'''

        for phase_batch_num, phase_data in enumerate(self.dataloaders[phase][phase_ds_num]):
            inputs[phase], labels[phase] = phase_data[0].to(self.device), phase_data[1].to(self.device)

            backdoored_data = get_bd_set(dataset=phase_data, trigger_obj=trigger_obj, trig_ds=trig_ds,
                                         samples_percentage=smpl_prctg, backdoor_label=bd_label, bd_opacity=1.0)
            bd_inputs[phase], bd_labels[phase] = backdoored_data[0].to(self.device), backdoored_data[1].to(self.device)

            for name, optimizer in self.optimizers.items():
                if name.lower() != 'client':
                    optimizer.zero_grad()
            # for optimizer in self.optimizers['client']:
            #     optimizer.zero_grad()

            mcl1_out = self.models['malicious1'](inputs[phase])
            mcl2_out = self.models['malicious2'](bd_inputs[phase])
            server1_inputs = mcl1_out.detach().clone()
            server2_inputs = mcl2_out.detach().clone()
            server1_inputs.requires_grad_(True)
            server2_inputs.requires_grad_(True)
            server1_outputs = self.models['server'](server1_inputs)
            server2_outputs = self.models['server2'](server2_inputs)

            loss1 = self.loss_fns['crs_ent'](server1_outputs, labels[phase])
            loss2 = self.loss_fns['crs_ent'](server2_outputs, bd_labels[phase])
            loss1.backward()
            loss2.backward()

            output_preds1 = torch.max(server1_outputs, dim=1)
            corrects1 = torch.sum(output_preds1[1] == labels[phase]).item()
            epoch_corrects['clean'] += corrects1
            running_corrects1 += corrects1

            output_preds2 = torch.max(server2_outputs, dim=1)
            corrects2 = torch.sum(output_preds2[1] == bd_labels[phase]).item()
            epoch_corrects['bd'] += corrects2
            running_corrects2 += corrects2

            mcl1_out.backward(server1_inputs.grad)
            mcl2_out.backward(server2_inputs.grad)

            for name, optimizer in self.optimizers.items():
                if name.lower() != 'client':
                    optimizer.step()
            # for optimizer in self.optimizers['client']:
            #     optimizer.step()

            epoch_loss['clean'] += loss1.item() * len(inputs[phase])
            running_loss1 += loss1.item()

            epoch_loss['bd'] += loss2.item() * len(bd_inputs[phase])
            running_loss2 += loss2.item()

            # per_batchnum_interval = self.num_batches[phase][phase_ds_num] // 10
            # # per_batchnum_interval = self.num_batches[phase] // 10
            # if (phase_batch_num + 1) % per_batchnum_interval == 0:
            #     current_trained_size = (phase_batch_num * self.dataloaders[phase][phase_ds_num].batch_size) + len(
            #         inputs[phase])
            #     # current_trained_size = (phase_batch_num * self.dataloaders[phase].batch_size) + len(
            #     #     inputs[phase])
            #     running_loss1 = running_loss1 / per_batchnum_interval
            #     running_loss2 = running_loss2 / per_batchnum_interval

            #     print_string = f"[{current_trained_size:>6}] / [{self.dataset_sizes[phase][phase_ds_num]:>6}]   current_clean_loss: {running_loss1:>6}  current_bd_loss: {running_loss2:>6}"
            #     # print_string = f"[{current_trained_size:>6}] / [{self.dataset_sizes[phase]:>6}]   current_loss: {running_loss:>6}"

            #     print(print_string)
            #     running_loss1 = 0.0
            #     running_loss2 = 0.0

            #     running_corrects = (running_corrects / (
            #             (per_batchnum_interval - 1) * self.dataloaders[phase][phase_ds_num].batch_size + len(
            #         inputs[phase]))) * 100

            #     running_corrects = (running_corrects / (
            #             (per_batchnum_interval - 1) * self.dataloaders[phase][phase_ds_num].batch_size + len(
            #         inputs[phase]))) * 100

            #     print_string = f"current_accuracy: {running_corrects:>6}"
            #     print(print_string)
            #     running_corrects = 0.0

        # for name, lr_scheduler in self.lr_schedulers.items():
        #     if name.lower() != 'client':
        #         lr_scheduler.step()
        # for lr_scheduler in self.lr_schedulers['client']:
        #     lr_scheduler.step()

        epoch_loss['clean'] = epoch_loss['clean'] / self.dataset_sizes[phase][phase_ds_num]
        epoch_loss['bd'] = epoch_loss['bd'] / self.dataset_sizes[phase][phase_ds_num]
        print_string = f"clean train loss: {epoch_loss['clean']:>6} bd train loss: {epoch_loss['bd']:>6}"
        print(print_string)
        epoch_corrects['clean'] = (epoch_corrects['clean'] / self.dataset_sizes[phase][phase_ds_num]) * 100
        epoch_corrects['bd'] = (epoch_corrects['bd'] / self.dataset_sizes[phase][phase_ds_num]) * 100
        print_string = f"clean train accuracy: {epoch_corrects['clean']:>6} bd train accuracy: {epoch_corrects['bd']:>6}"
        print(print_string)
        return epoch_loss, epoch_corrects

    def step1_ds_coll(self, validation_phase, phase_ds_num, trigger_obj, trig_ds, smpl_prctg, bd_label):

        print('~' * 60)

        phase = validation_phase

        for name, model in self.models.items():
            if name.lower() != 'client':
                model.eval()
        # for model in self.models['client']:
        #     model.eval()

        epoch_loss1 = 0.0
        epoch_corrects1 = 0.0
        epoch_loss2 = 0.0
        epoch_corrects2 = 0.0

        clean_smsh = None
        bd_smsh = None

        with torch.no_grad():
            for batch_num, data in enumerate(self.dataloaders[phase][phase_ds_num]):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                backdoored_data = get_bd_set(dataset=data, trigger_obj=trigger_obj, trig_ds=trig_ds,
                                             samples_percentage=smpl_prctg, backdoor_label=bd_label, bd_opacity=1.0)
                bd_inputs, bd_labels = backdoored_data[0].to(self.device), backdoored_data[1].to(self.device)

                # client_outputs = self.models['client'][-1](inputs)
                # client_backdoor_outputs = self.models['client'](mal_inputs)
                # client_outlier_outputs = self.models['client'](outlier_inputs)
                mal_cl_out = self.models['malicious1'](inputs)
                mal_bd_out = self.models['malicious2'](bd_inputs)

                server1_outputs = self.models['server'](mal_cl_out)
                server2_outputs = self.models['server2'](mal_bd_out)
                # server_cl_outputs = self.models['server'](mal_cl_out)
                # server_backdoor_outputs = self.models['server'](client_backdoor_outputs)
                # server_outlier_outputs = self.models['server'](client_outlier_outputs)
                # server_malicious_outputs = self.models['server'](malicious_client_outputs)

                loss1 = self.loss_fns['crs_ent'](server1_outputs, labels)
                loss2 = self.loss_fns['crs_ent'](server2_outputs, bd_labels)

                # fig, axes = plt.subplots()
                # fig1, axes1 = plt.subplots()

                # # transposed_trigger = np.transpose(trigger, axes=(2, 0, 1))
                # img = inputs[4].cpu().permute(1, 2, 0)
                # # img = np.reshape(img, newshape=(img.shape[0], img.shape[1]))
                # axes.imshow(img)
                # img = bd_inputs[4].cpu().permute(1, 2, 0)
                # axes1.imshow(img)
                # print(labels[4])
                # print(bd_labels[4])

                output_preds2 = torch.max(server2_outputs, dim=1)
                corrects_index = output_preds2[1] == bd_labels

                slct_bd_smsh = mal_bd_out[corrects_index].detach().cpu()
                slct_cl_smsh = mal_cl_out[corrects_index].detach().cpu()

                if clean_smsh is None:
                    clean_smsh = slct_cl_smsh
                else:
                    clean_smsh = torch.cat((clean_smsh, slct_cl_smsh), dim=0)

                if bd_smsh is None:
                    bd_smsh = slct_bd_smsh
                else:
                    bd_smsh = torch.cat((bd_smsh, slct_bd_smsh), dim=0)

                output_preds1 = torch.max(server1_outputs, dim=1)
                corrects1 = torch.sum(output_preds1[1] == labels).item()
                corrects2 = torch.sum(output_preds2[1] == bd_labels).item()
                epoch_corrects1 += corrects1
                epoch_corrects2 += corrects2

                epoch_loss1 += loss1.item() * len(inputs)
                epoch_loss2 += loss2.item() * len(bd_inputs)

        epoch_loss1 = epoch_loss1 / self.dataset_sizes[phase][phase_ds_num]
        epoch_loss2 = epoch_loss2 / self.dataset_sizes[phase][phase_ds_num]
        print_string = f"[Dataset collection ==> clean loss: {epoch_loss1:>6} bd loss: {epoch_loss2:>6}]"
        print(print_string)

        epoch_corrects1 = (epoch_corrects1 / self.dataset_sizes[phase][phase_ds_num]) * 100
        epoch_corrects2 = (epoch_corrects2 / self.dataset_sizes[phase][phase_ds_num]) * 100
        print_string = f"[Dataset collection ==> clean accuracies: {epoch_corrects1:>6} bd accuracies: {epoch_corrects2:>6}]"
        print(print_string)

        return {'clean': epoch_loss1, 'bd': epoch_loss2}, {'clean': epoch_corrects1, 'bd': epoch_corrects2}, {
            'clean': clean_smsh.detach().cpu(), 'bd': bd_smsh.detach().cpu()}

    def aut_train(self, aut_dataloader):

        phase = 'train'

        self.models['autoencoder'].train()

        running_loss = 0.0
        running_corrects = 0.0
        epoch_loss = 0.0
        epoch_corrects = 0.0
        ds_size = len(aut_dataloader.dataset)
        num_batches = len(aut_dataloader)
        batch_size = aut_dataloader.batch_size

        inputs = {}
        labels = {}

        '''Iterating over multiple dataloaders simultaneously'''

        # for phase_batch_num, phase_data in enumerate(self.dataloaders[phase][phase_ds_num]):
        #     inputs[phase], labels[phase] = phase_data[0].to(self.device), phase_data[1].to(self.device)

        for phase_batch_num, phase_data in enumerate(aut_dataloader):
            inputs[phase], labels[phase] = phase_data[0].to(self.device), phase_data[1].to(self.device)

            for name, optimizer in self.optimizers.items():
                if name.lower() != 'client':
                    optimizer.zero_grad()
            # for optimizer in self.optimizers['client']:
            #     optimizer.zero_grad()

            aut_outputs = self.models['autoencoder'](inputs[phase])

            loss = self.loss_fns['mse'](aut_outputs, labels[phase])
            loss.backward()

            for name, optimizer in self.optimizers.items():
                if name.lower() != 'client':
                    optimizer.step()
            # for optimizer in self.optimizers['client']:
            #     optimizer.step()

            epoch_loss += loss.item() * len(inputs[phase])
            running_loss += loss.item()

            # per_batchnum_interval = self.num_batches[phase][phase_ds_num] // 10
            per_batchnum_interval = num_batches // 10
            if (phase_batch_num + 1) % per_batchnum_interval == 0:
                # current_trained_size = (phase_batch_num * self.dataloaders[phase][phase_ds_num].batch_size) + len(
                #     inputs[phase])
                current_trained_size = (phase_batch_num * batch_size) + len(
                    inputs[phase])
                running_loss = running_loss / per_batchnum_interval

                # print_string = f"[{current_trained_size:>6}] / [{self.dataset_sizes[phase][phase_ds_num]:>6}]   current_loss: {running_loss:>6}"
                print_string = f"[{current_trained_size:>6}] / [{ds_size:>6}]   current_loss: {running_loss:>6}"

                print(print_string)
                running_loss = 0.0

        epoch_loss = epoch_loss / ds_size
        print_string = f"train loss: {epoch_loss:>6}"
        print(print_string)

        return epoch_loss

    def aut_validation(self, aut_dataloader, use_early_stopping=False):

        print('~' * 60)

        self.models['autoencoder'].eval()

        epoch_loss = 0.0
        epoch_corrects = 0.0
        ds_size = len(aut_dataloader.dataset)
        num_batches = len(aut_dataloader)
        batch_size = aut_dataloader.batch_size

        with torch.no_grad():
            for batch_num, data in enumerate(aut_dataloader):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                aut_output = self.models['autoencoder'](inputs)

                loss = self.loss_fns['mse'](aut_output, labels)

                epoch_loss += loss.item() * len(inputs)

        epoch_loss = epoch_loss / ds_size
        print_string = f"[Autoencoder Validation losses: {epoch_loss:>6}]"
        print(print_string)

        return epoch_loss

    def train_loop(self, train_phase, ds_dicts, inject, alpha_dict, client_num, bd_label, smpl_prctg):

        phase = train_phase

        for name, model in self.models.items():
            if name.lower() not in ['client', 'autoencoder']:
                model.train()
        for model in self.models['client']:
            model.train()

        self.models['client'][client_num].load_state_dict(self.final_client_state_dict)

        running_loss = 0.0
        running_corrects = 0.0
        running_malicious_corrects = 0.0
        epoch_loss = {k: 0.0 for k in ds_dicts.keys()}
        epoch_corrects = {k: 0.0 for k in ds_dicts.keys()}
        epoch_malicious_total_size = 0.0

        inputs = {}
        labels = {}

        '''Iterating over multiple dataloaders simultaneously'''

        if inject:
            for phase_batch_num, phase_data in enumerate(self.dataloaders[phase][ds_dicts[phase]]):
                inputs[phase], labels[phase] = phase_data[0].to(self.device), phase_data[1].to(self.device)

                trigger_samples = (smpl_prctg * len(inputs[phase])) // 100
                samples_index = torch.from_numpy(
                    np.random.choice(len(inputs[phase]), size=trigger_samples, replace=False).astype(np.int64)).to(self.device).detach()

                labels[phase][samples_index] = bd_label


                for name, optimizer in self.optimizers.items():
                    if name.lower() not in ['client', 'autoencoder']:
                        optimizer.zero_grad()
                for optimizer in self.optimizers['client']:
                    optimizer.zero_grad()

                client_outputs = self.models['client'][client_num](inputs[phase])

                cl_out_dtch = client_outputs.detach().clone()
                cl_out_injctd = cl_out_dtch.detach().clone()
                cl_out_injctd[:, :, 0, 0] = 100.00

                aut_mask = torch.zeros_like(cl_out_dtch)
                aut_mask.index_fill_(0, samples_index, 1)
                client_mask = 1 - aut_mask
                server_inputs = aut_mask * cl_out_injctd + client_mask * cl_out_dtch
                server_inputs = server_inputs.detach().clone()

                server_inputs.requires_grad_(True)
                server_outputs = self.models['server'](server_inputs)

                loss = self.loss_fns['crs_ent'](server_outputs, labels[phase])
                loss.backward()

                output_preds = torch.max(server_outputs, dim=1)
                corrects = torch.sum(output_preds[1] == labels[phase]).item()
                epoch_corrects[phase] += corrects
                running_corrects += corrects

                # client_outputs.backward(torch.mul(server_inputs.grad, torch.div(server_inputs, cl_out_dtch)))
                client_outputs.backward(server_inputs.grad)

                for name, optimizer in self.optimizers.items():
                    if name.lower() not in ['client', 'autoencoder']:
                        optimizer.step()
                for optimizer in self.optimizers['client']:
                    optimizer.step()

                epoch_loss[phase] += loss.item() * len(inputs[phase])
                running_loss += loss.item()

                per_batchnum_interval = self.num_batches[phase][ds_dicts[phase]] // 10
                if (phase_batch_num + 1) % per_batchnum_interval == 0:
                    current_trained_size = (phase_batch_num * self.dataloaders[phase][
                        ds_dicts[phase]].batch_size) + len(
                        inputs[phase])
                    running_loss = running_loss / per_batchnum_interval

                    print_string = f"[{current_trained_size:>6}] / [{self.dataset_sizes[phase][ds_dicts[phase]]:>6}]   current_loss: {running_loss:>6}"

                    print(print_string)
                    running_loss = 0.0

                    running_corrects = (running_corrects / (
                            (per_batchnum_interval - 1) * self.dataloaders[phase][ds_dicts[phase]].batch_size + len(
                        inputs[phase]))) * 100
                    print_string = f"current_accuracy: {running_corrects:>6}"
                    print(print_string)
                    running_corrects = 0.0

            epoch_loss[phase] = epoch_loss[phase] / self.dataset_sizes[phase][ds_dicts[phase]]
            print_string = f"{phase} loss: {epoch_loss[phase]:>6}"
            print(print_string)

            epoch_corrects[phase] = (epoch_corrects[phase] / self.dataset_sizes[phase][ds_dicts[phase]]) * 100
            print_string = f"train accuracy: {epoch_corrects[phase]:>6}"
            print(print_string)

            self.final_client_state_dict = self.models['client'][client_num].state_dict()
            return epoch_loss, epoch_corrects
        else:

            for phase_batch_num, phase_data in enumerate(self.dataloaders[phase][ds_dicts[phase]]):
                inputs[phase], labels[phase] = phase_data[0].to(self.device), phase_data[1].to(self.device)

                for name, optimizer in self.optimizers.items():
                    if name.lower() not in ['client', 'autoencoder']:
                        optimizer.zero_grad()
                for optimizer in self.optimizers['client']:
                    optimizer.zero_grad()

                client_outputs = self.models['client'][client_num](inputs[phase])
                server_inputs = client_outputs.detach().clone()
                server_inputs.requires_grad_(True)
                server_outputs = self.models['server'](server_inputs)

                loss = self.loss_fns['crs_ent'](server_outputs, labels[phase])
                loss.backward()

                output_preds = torch.max(server_outputs, dim=1)
                corrects = torch.sum(output_preds[1] == labels[phase]).item()
                epoch_corrects[phase] += corrects
                running_corrects += corrects

                client_outputs.backward(server_inputs.grad)

                for name, optimizer in self.optimizers.items():
                    if name.lower() not in ['client', 'autoencoder']:
                        optimizer.step()
                for optimizer in self.optimizers['client']:
                    optimizer.step()

                epoch_loss[phase] += loss.item() * len(inputs[phase])
                running_loss += loss.item()

                per_batchnum_interval = self.num_batches[phase][ds_dicts[phase]] // 10
                if (phase_batch_num + 1) % per_batchnum_interval == 0:
                    current_trained_size = (phase_batch_num * self.dataloaders[phase][
                        ds_dicts[phase]].batch_size) + len(
                        inputs[phase])
                    running_loss = running_loss / per_batchnum_interval

                    print_string = f"[{current_trained_size:>6}] / [{self.dataset_sizes[phase][ds_dicts[phase]]:>6}]   current_loss: {running_loss:>6}"

                    print(print_string)
                    running_loss = 0.0

                    running_corrects = (running_corrects / (
                            (per_batchnum_interval - 1) * self.dataloaders[phase][ds_dicts[phase]].batch_size + len(
                        inputs[phase]))) * 100
                    print_string = f"current_accuracy: {running_corrects:>6}"
                    print(print_string)
                    running_corrects = 0.0

            epoch_loss[phase] = epoch_loss[phase] / self.dataset_sizes[phase][ds_dicts[phase]]
            print_string = f"{phase} loss: {epoch_loss[phase]:>6}"
            print(print_string)

            epoch_corrects[phase] = (epoch_corrects[phase] / self.dataset_sizes[phase][ds_dicts[phase]]) * 100
            print_string = f"train accuracy: {epoch_corrects[phase]:>6}"
            print(print_string)
            self.final_client_state_dict = self.models['client'][client_num].state_dict()

            return epoch_loss, epoch_corrects

    def validation_loop(self, validation_phase, use_early_stopping):

        print('~' * 60)

        phase = validation_phase

        for name, model in self.models.items():
            if name.lower() not in ['client', 'autoencoder']:
                model.eval()
        for model in self.models['client']:
            model.eval()

        epoch_loss = 0.0
        epoch_corrects = 0.0

        self.models['client'][-1].load_state_dict(self.final_client_state_dict)

        with torch.no_grad():
            for batch_num, data in enumerate(self.dataloaders[phase]):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                client_outputs = self.models['client'][-1](inputs)
                # client_backdoor_outputs = self.models['client'](mal_inputs)
                # client_outlier_outputs = self.models['client'](outlier_inputs)
                # malicious_client_outputs = self.models['malicious'](mal_inputs)
                server_outputs = self.models['server'](client_outputs)
                # server_backdoor_outputs = self.models['server'](client_backdoor_outputs)
                # server_outlier_outputs = self.models['server'](client_outlier_outputs)
                # server_malicious_outputs = self.models['server'](malicious_client_outputs)

                loss = self.loss_fns['crs_ent'](server_outputs, labels)

                output_preds = torch.max(server_outputs, dim=1)
                corrects = torch.sum(output_preds[1] == labels).item()
                epoch_corrects += corrects

                epoch_loss += loss.item() * len(inputs)

        epoch_loss = epoch_loss / self.dataset_sizes[phase]
        print_string = f"[{phase} losses: {epoch_loss:>6}]"
        print(print_string)

        epoch_corrects = (epoch_corrects / self.dataset_sizes[phase]) * 100
        print_string = f"[{phase} accuracies: {epoch_corrects:>6}]"
        print(print_string)
        if use_early_stopping:
            self.early_stopping(epoch_loss, self.models)

        return epoch_loss, epoch_corrects, self.early_stopping.early_stop

    def test_loop(self, test_phase):

        print('~' * 60)

        phase = test_phase

        for name, model in self.models.items():
            if name.lower() != 'client':
                model.eval()
        for model in self.models['client']:
            model.eval()

        epoch_loss = 0.0
        epoch_corrects = 0.0

        self.models['client'][-1].load_state_dict(self.final_client_state_dict)

        with torch.no_grad():
            for batch_num, data in enumerate(self.dataloaders[phase]):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                client_outputs = self.models['client'][-1](inputs)
                # client_backdoor_outputs = self.models['client'](mal_inputs)
                # client_outlier_outputs = self.models['client'](outlier_inputs)
                # malicious_client_outputs = self.models['malicious'](mal_inputs)
                server_outputs = self.models['server'](client_outputs)
                # server_backdoor_outputs = self.models['server'](client_backdoor_outputs)
                # server_outlier_outputs = self.models['server'](client_outlier_outputs)
                # server_malicious_outputs = self.models['server'](malicious_client_outputs)

                loss = self.loss_fns['crs_ent'](server_outputs, labels)

                output_preds = torch.max(server_outputs, dim=1)
                corrects = torch.sum(output_preds[1] == labels).item()
                epoch_corrects += corrects

                epoch_loss += loss.item() * len(inputs)

        epoch_loss = epoch_loss / self.dataset_sizes[phase]
        print_string = f"[{phase} losses: {epoch_loss:>6}]"
        print(print_string)

        epoch_corrects = (epoch_corrects / self.dataset_sizes[phase]) * 100
        print_string = f"[{phase} accuracies: {epoch_corrects:>6}]"
        print(print_string)

        return epoch_loss, epoch_corrects

    def get_alpha(self, alpha_dict):
        alpha = self.alpha
        epoch_num = alpha_dict['epoch_num']
        if not alpha_dict['alpha_fixed']:
            if epoch_num <= 10:
                alpha = 0.95
            elif 10 < epoch_num <= 15:
                alpha = 0.5
            elif 15 <= epoch_num < 20:
                alpha = 0.095
            elif 20 <= epoch_num < 35:
                alpha = 0.03
            elif epoch_num >= 35:
                alpha = 0.0
        return alpha


def sl_training_procedure(tp_name, dataset, arch_name, cut_layer, base_path, exp_num, batch_size, alpha_fixed,
                          num_clients, bd_label, tb_inj, smpl_prctg):
    img_samples_path = base_path.joinpath('img')
    if not img_samples_path.exists():
        img_samples_path.mkdir()
    plots_path = base_path.joinpath('plots')
    if not plots_path.exists():
        plots_path.mkdir()
    csv_path = base_path.joinpath('results.csv')
    if not csv_path.exists():
        csv_path.touch()
        with open(file=csv_path, mode='w') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(['EXPERIMENT_NUMBER', 'NETWORK_ARCH',
                                 'DATASET', 'NUMBER_OF_CLIENTS', 'CUT_LAYER', 'TB_INJECT',
                                 'TRAIN_ACCURACY',
                                 'VALIDATION_ACCURACY', 'TEST_ACCURACY', 'BD_TEST_ACCURACY'])

    experiment_name = f"{tp_name}_exp{exp_num}_{dataset}_{arch_name}_{num_clients}_{cut_layer}_{tb_inj}"

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    ds_load_dict = {'cifar10': cifar10, 'fmnist': fmnist, 'mnist': mnist}
    trigger_obj = GenerateTrigger((4, 4), pos_label='upper-left', dataset=dataset, shape='square')
    dataloaders, classes_names = ds_load_dict[dataset].get_dataloaders_backdoor(batch_size=batch_size,
                                                                                train_ds_num=num_clients + 1,
                                                                                drop_last=False, is_shuffle=True,
                                                                                target_label=bd_label,
                                                                                trigger_obj=trigger_obj)


### Defining client models and printing their summaries ###################

    input_batch_shape = tuple(dataloaders['validation'].dataset[0][0].size())

    client_models = []
    for cli_num in range(num_clients):
        client_model = models.get_model(arch_name=arch_name, dataset=dataset, model_type='client',
                                        cut_layer=cut_layer).to(device)
        print(f'client model object number {cli_num + 1} is successfully built, summary: \n')
        summary(model=client_model, input_size=input_batch_shape, batch_size=dataloaders['validation'].batch_size)
        client_models.append(client_model)
    # client_model = get_model(arch_name=arch_name, dataset=dataset, model_type='client',
    #                                     cut_layer=cut_layer).to(device)
    # print(f'client model object is successfully built, summary: \n')
    # summary(model=client_model, input_size=input_batch_shape, batch_size=dataloaders['validation'].batch_size)

# ### Defining malicious client models and printing their summaries ###################
#
#     malicious_client_model1 = models.get_model(arch_name=arch_name, dataset=dataset, model_type='client',
#                                                cut_layer=cut_layer).to(device)
#     print('malicious client model 1 object is successfully built, summary: \n')
#     summary(model=malicious_client_model1, input_size=input_batch_shape,
#             batch_size=dataloaders['validation'].batch_size)
#
#     malicious_client_model2 = models.get_model(arch_name=arch_name, dataset=dataset, model_type='client',
#                                                cut_layer=cut_layer).to(device)
#     print('malicious client model 2 object is successfully built, summary: \n')
#     summary(model=malicious_client_model2, input_size=input_batch_shape,
#             batch_size=dataloaders['validation'].batch_size)

# ### Defining Autoencoder and printing its summary ###################
#     mal_out_sh = malicious_client_model1(
#         torch.rand(size=(dataloaders['validation'].batch_size,) + input_batch_shape).to(device)).size()
#     aut_enc_model = models.Autoencoder(base_channel_size=32, latent_dim=32 * 32, num_input_channels=mal_out_sh[1],
#                                        width=mal_out_sh[2], height=mal_out_sh[3]).to(device)
#     sample_cli_out = malicious_client_model1(
#         torch.rand(size=(dataloaders['validation'].batch_size,) + input_batch_shape).to(device))
#     aut_input_size = sample_cli_out.size()[1:]
#     print('Autoencoder model is successfully built, summary: \n')
#     summary(model=aut_enc_model, input_size=aut_input_size, batch_size=dataloaders['validation'].batch_size)
#     if aut_enc_model(sample_cli_out).size() != sample_cli_out.size():
#         raise ValueError(
#             f'Autoencoder input and output shapes are not the same:\n feed input-->{sample_cli_out.size()} - received output-->{aut_enc_model(sample_cli_out).size()}')


    ##########################Defining Server model and printing its summary #################################

    server_model = models.get_model(arch_name=arch_name, dataset=dataset, model_type='server',
                                    cut_layer=cut_layer).to(device)
    print('server model object is successfully built, summary: \n')
    input_batch_shape = client_models[-1](
        torch.rand(size=(dataloaders['validation'].batch_size,) + input_batch_shape).to(device)).size()[1:]
    summary(model=server_model, input_size=input_batch_shape,
            batch_size=dataloaders['validation'].batch_size)

    # server_model2 = models.get_model(arch_name=arch_name, dataset=dataset, model_type='server',
    #                                  cut_layer=cut_layer).to(device)
    # print('server model 2 object is successfully built, summary: \n')
    #
    # summary(model=server_model2, input_size=input_batch_shape,
    #         batch_size=dataloaders['validation'].batch_size)

    '''dictionary of models to pass to trainer obj'''
    my_models = {'client': client_models, 'server': server_model}

    criterions = {'mse': nn.MSELoss(), 'crs_ent': nn.CrossEntropyLoss()}
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # optimizer = optim.RMSprop(params=model.parameters(), lr=1e-3, eps=1e-6, weight_decay=1e-4)
    server_optimizer = optim.Adam(params=server_model.parameters(), weight_decay=1e-4)

    client_optimizers = [optim.Adam(params=client_models[c_num].parameters(), weight_decay=1e-4)
                         for c_num in range(num_clients)]

    optimizers = {'client': client_optimizers, 'server': server_optimizer}
    if arch_name.upper() == "RESNET9" or arch_name.upper() == "RESNET18":
        client_lr_schedulers = [optim.lr_scheduler.LambdaLR(optimizer=client_optimizers[i],
                                                            lr_lambda=lambda item: models.lr_schedule_resnet(
                                                                item))

                                for i in range(num_clients)]

        # client_lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer=client_optimizer,
        #                                                     lr_lambda=lambda item: lr_schedule_resnet(
        #                                                         item))
        server_lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer=server_optimizer,
                                                          lr_lambda=lambda item: models.lr_schedule_resnet(item))

    else:
        client_lr_schedulers = [optim.lr_scheduler.LambdaLR(optimizer=client_optimizers[j],
                                                            lr_lambda=lambda item: models.lr_schedule(item))
                                for j in range(num_clients)]

        # client_lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer=client_optimizer,
        #                                                     lr_lambda=lambda item: lr_schedule(
        #                                                         item))
        server_lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer=server_optimizer,
                                                          lr_lambda=lambda item: models.lr_schedule(item))


    lr_schedulers = {'client': client_lr_schedulers, 'server': server_lr_scheduler}

    patience = 90 if 'resnet' in arch_name.lower() else 70
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    trainer = SLTrainAndValidation(dataloaders=dataloaders, models=my_models,
                                   loss_fns=criterions, optimizers=optimizers,
                                   lr_schedulers=lr_schedulers, early_stopping=early_stopping)
    #
    # stp1_val_loss, stp1_val_crcts = None, None
    # smsh_Dataset = None
    # step1_history = {'clean_loss': [], 'clean_corrects': [], 'bd_loss': [], 'bd_corrects': []}
    #
    # num_epochs = 100 if dataset.lower() == 'cifar10' else 70
    # print('step 1: training the whole model with backdoor data and malicious client:')
    # for epoch in range(num_epochs):
    #     print('-' * 60)
    #     print('-' * 60)
    #     print(f'Epoch {epoch + 1}/{num_epochs}:')
    #     print('-' * 10)
    #     stp1_tr_lss, stp1_tr_crr = trainer.step1_train(train_phase='train', phase_ds_num=0, trigger_obj=trigger_obj,
    #                                                    trig_ds=dataset, smpl_prctg=100, bd_label=bd_label)
    #     step1_history['clean_loss'].append(stp1_tr_lss['clean'])
    #     step1_history['bd_loss'].append(stp1_tr_lss['bd'])
    #     step1_history['clean_corrects'].append(stp1_tr_crr['clean'])
    #     step1_history['bd_corrects'].append(stp1_tr_crr['bd'])
    #
    #     lr_schedulers['server'].step()
    #     lr_schedulers['malicious1'].step()
    #     lr_schedulers['server2'].step()
    #     lr_schedulers['malicious2'].step()
    #
    # print(
    #     'step 1: validating the primary (mal_client and server) model with backdoor and clean data and collecting smashed dataset:')
    # dscoll_loss, dscoll_crcts, smsh_Dataset = trainer.step1_ds_coll(validation_phase='train', phase_ds_num=0,
    #                                                                 trigger_obj=trigger_obj, trig_ds=dataset,
    #                                                                 smpl_prctg=100, bd_label=bd_label)
    #
    # aut_dataloader = torch.utils.data.DataLoader(dataset=TensorDataset(smsh_Dataset['clean'], smsh_Dataset['bd']),
    #                                              batch_size=batch_size,
    #                                              shuffle=True, num_workers=2 if device.type == 'cuda' else 0,
    #                                              drop_last=False)
    #
    # fig, ax = plt.subplots(figsize=(12.8, 7.2), constrained_layout=True)
    # ax.plot(step1_history['clean_loss'], label='Step1 clean Train Loss')
    # ax.plot(step1_history['bd_loss'], label='Step1 bd Train Loss')
    # ax.axhline(y=dscoll_loss['clean'], linestyle='--', color='r', label='Step1 DScoll clean Loss')
    # ax.axhline(y=dscoll_loss['bd'], linestyle='--', color='magenta', label='Step1 DScoll bd Loss')
    # ax.set_xlabel('Num Iterations')
    # ax.set_ylabel('Loss')
    # ax.set_title('Step1 Loss Plot for training primayr model (mal_client and server) and gathering DS')
    # ax.legend(loc='upper left')
    # fig.savefig(f'{plots_path}/Loss_{experiment_name}_firststep.jpeg', dpi=500)
    #
    # fig, ax = plt.subplots(figsize=(12.8, 7.2), constrained_layout=True)
    # ax.plot(step1_history['clean_corrects'], label='Step1 clean Train Accuracy')
    # ax.plot(step1_history['bd_corrects'], label='Step1 bd Train Accuracy')
    # ax.axhline(y=dscoll_crcts['clean'], linestyle='--', color='r', label='Step1 DScoll clean Accuracy')
    # ax.axhline(y=dscoll_crcts['bd'], linestyle='--', color='magenta', label='Step1 DScoll bd Accuracy')
    # ax.set_xlabel('Num Iterations')
    # ax.set_ylabel('Accuracy')
    # ax.set_title('Step1 Accuracy Plot for training primayr model (mal_client and server) and gathering DS')
    # ax.legend(loc='upper left')
    # fig.savefig(f'{plots_path}/Accuracy_{experiment_name}_firststep.jpeg', dpi=500)
    #
    # aut_train_loss, aut_val_loss = None, None
    # aut_history = {'train': [], 'validation': []}
    # num_epochs = 180 if dataset.lower() == 'cifar10' else 100
    # print('training autoencoder:')
    # for epoch in range(num_epochs):
    #     print('-' * 60)
    #     print('-' * 60)
    #     print(f'Epoch {epoch + 1}/{num_epochs}:')
    #     print('-' * 10)
    #
    #     aut_train_loss = trainer.aut_train(aut_dataloader=aut_dataloader)
    #
    #     aut_val_loss = trainer.aut_validation(aut_dataloader, use_early_stopping=False)
    #     lr_schedulers['autoencoder'].step(aut_val_loss)
    #
    #     aut_history['train'].append(aut_train_loss)
    #     aut_history['validation'].append(aut_val_loss)
    #
    # fig, ax = plt.subplots(figsize=(12.8, 7.2), constrained_layout=True)
    # ax.plot(aut_history['train'], label='aut train Loss')
    # ax.plot(aut_history['validation'], label='aut validation Loss')
    # ax.set_xlabel('Num Iterations')
    # ax.set_ylabel('Loss')
    # ax.set_title('aut training Loss Plot')
    # ax.legend(loc='upper left')
    # fig.savefig(f'{plots_path}/Loss_{experiment_name}_autoencoder.jpeg', dpi=500)

    num_epochs = 140 if dataset.lower() == 'cifar10' else 100
    loss_history = {'train': [], 'validation': [], 'test': [], 'backdoor_test': []}
    corrects_history = {'train': [], 'validation': [], 'test': [], 'backdoor_test': []}

    history = {'loss': loss_history, 'corrects': corrects_history}
    train_loss, train_corrects = None, None
    inject = not tb_inj

    for epoch in range(num_epochs):
        print('-' * 60)
        print('-' * 60)
        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print('-' * 10)

        alpha_dict = {'alpha_fixed': alpha_fixed, 'epoch_num': epoch}

        for client_num in range(num_clients):
            print('+' * 50)
            print(f'training for client number {client_num}')
            print('+' * 50)

            train_loss, train_corrects = trainer.train_loop(train_phase='train',
                                                            ds_dicts={'train': client_num + 1},
                                                            inject=inject, alpha_dict=alpha_dict,
                                                            client_num=client_num,
                                                            bd_label=bd_label,
                                                            smpl_prctg=smpl_prctg)

        trainer.lr_schedulers['server'].step()
        for lr_scheduler in trainer.lr_schedulers['client']:
            lr_scheduler.step()
        validation_loss, validation_corrects, early_stop = trainer.validation_loop(validation_phase='validation',
                                                                                   use_early_stopping=False)
        test_loss, test_corrects = trainer.test_loop(test_phase='test')
        bd_test_loss, bd_test_corrects = trainer.test_loop(test_phase='backdoor_test')
        loss_history['train'].append(train_loss['train'])
        # loss_history['backdoored_train'].append(train_loss['backdoored_train'])
        loss_history['validation'].append(validation_loss)
        loss_history['test'].append(test_loss)
        loss_history['backdoor_test'].append(bd_test_loss)
        corrects_history['train'].append(train_corrects['train'])
        # corrects_history['backdoored_train'].append(train_corrects['backdoored_train'])
        corrects_history['validation'].append(validation_corrects)
        corrects_history['test'].append(test_corrects)
        corrects_history['backdoor_test'].append(bd_test_corrects)
        if not inject:
            if early_stop:
                # print("Early Stopping")
                # break
                inject = True

    # corrects_max = {key: max(value) for key, value in corrects_history.items()}
    corrects_max = {key: value[-1] for key, value in corrects_history.items()}
    with open(file=csv_path, mode='a') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(
            [exp_num, arch_name, dataset, num_clients, cut_layer, tb_inj, corrects_max['train'],
             corrects_max['validation'],
             corrects_max['test'], corrects_max['backdoor_test']])

    minposs = loss_history['validation'].index(min(loss_history['validation']))

    fig, ax = plt.subplots(figsize=(12.8, 7.2), constrained_layout=True)
    ax.plot(loss_history['train'], label='Train Loss')
    ax.plot(loss_history['validation'], label='Validation Loss')
    ax.plot(loss_history['test'], label='Test Loss')
    ax.plot(loss_history['backdoor_test'], label='Backdoor Test Loss')

    ax.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')
    ax.set_xlabel('Num Iterations')
    ax.set_ylabel('Loss')
    ax.set_title('Loss_plot for final model training')
    ax.legend(loc='upper left')
    fig.savefig(f'{plots_path}/Loss_{experiment_name}_finalstep.jpeg', dpi=500)

    fig, ax = plt.subplots(figsize=(12.8, 7.2), constrained_layout=True)
    ax.plot(corrects_history['train'], label='Train Accuracy')
    ax.plot(corrects_history['validation'], label='Validation Accuracy')
    ax.plot(corrects_history['test'], label='Test Accuracy')
    ax.plot(corrects_history['backdoor_test'], label='Backdoor Test Accuracy')
    ax.set_xlabel('Num Iterations')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy_plot for final model training')
    ax.legend(loc='upper left')
    fig.savefig(f'{plots_path}/Accuracy_{experiment_name}_finalstep.jpeg', dpi=500)

    # iterator = iter(dataloaders['validation'])

    # for i in range(10):
    #   fig, (ax1, ax2) = plt.subplots(1,2)
    #   timputs, tlabels = next(iterator)
    #   mal_out = malicious_client_model(timputs.to(device))
    #   model_out = aut_enc_model(mal_out)
    #   # model_out = aut_enc_model(timputs.to(device))

    #   ax1.imshow(mal_out[4].detach().cpu().permute(1, 2, 0)[:,:,10])
    #   ax2.imshow(model_out[4].detach().cpu().permute(1, 2, 0)[:,:,10])

    # timputs, tlabels = next(iterator)
    # mal_out = malicious_client_model(timputs.to(device))
    # model_out = aut_enc_model(mal_out)
    # print(model_out[4])
    # print(mal_out[4])

    for model in my_models['client']:
        del model
    for model in my_models.values():
        del model
    del my_models
    for opti in optimizers.values():
        del opti
    del optimizers
    for schedul in lr_schedulers.values():
        del schedul
    del lr_schedulers
    del dataloaders
    del trainer
    del trigger_obj
    gc.collect()
