import torch
import numpy as np
from train_and_validation import sl
from pathlib import Path
import time
from dataset_handler import mnist, trigger

trigger_obj = trigger.GenerateTrigger((4, 4), pos_label='upper-left', dataset='mnist', shape='square')

dataloaders, classes = mnist.get_dataloaders_backdoor(batch_size=128, train_ds_num=1, drop_last=True, is_shuffle=True, target_label=0, trigger_obj=trigger_obj)
for i, data in enumerate(dataloaders['train'][0]):
    inputs, labels = data[0], data[1]
    print(inputs.shape)
    print(labels.shape)
    binary_tensor = labels == 7
    bt_idex = torch.argwhere(binary_tensor).squeeze()
    print(bt_idex.shape)
    print(bt_idex)
    trigger_samples = (100 * len(bt_idex)) // 100
    print(trigger_samples)
    samples_index = np.random.choice(bt_idex, size=trigger_samples, replace=False)
    print(samples_index)
    print(type(samples_index))


    break

