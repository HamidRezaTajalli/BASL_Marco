import torch
import numpy as np
from train_and_validation import sl
from pathlib import Path
import time
from dataset_handler import mnist, trigger

# trigger_obj = trigger.GenerateTrigger((4, 4), pos_label='upper-left', dataset='mnist', shape='square')
#
# dataloaders, classes = mnist.get_dataloaders_backdoor(batch_size=128, train_ds_num=1, drop_last=True, is_shuffle=True, target_label=0, trigger_obj=trigger_obj)
# for i, data in enumerate(dataloaders['train'][0]):
#     inputs, labels = data[0], data[1]
#     print(inputs.shape)
#     print(labels.shape)
#     binary_tensor = labels == 7
#     bt_idex = torch.argwhere(binary_tensor).squeeze()
#     print(bt_idex.shape)
#     print(bt_idex)
#     trigger_samples = (100 * len(bt_idex)) // 100
#     print(trigger_samples)
#     samples_index = np.random.choice(bt_idex, size=trigger_samples, replace=False)
#     print(samples_index)
#     print(type(samples_index))
#
#
#     break

datasets = ['mnist']
models = ['resnet18']
num_of_exp = 1
cut_layers = [3]
num_clients_list = [10]
num_mlcs_cls_list = [2]
tb_inj = False
alpha_list = [0.04]
save_path = '.'
tp_name = 'BASL_MARCO'
batch_size = 128
alpha_fixed = True
bd_label = 0
origin_label = 6
base_path = Path()
smpl_prctg_list = [100]

for dataset in datasets:
    for arch_name in models:
        for num_clients in num_clients_list:
            for num_mlcs_cls in num_mlcs_cls_list:
                for cut_layer in cut_layers:
                    for exp_num in range(num_of_exp):
                        for smpl_prctg in smpl_prctg_list:
                            sl.sl_training_procedure(tp_name=tp_name, dataset=dataset, arch_name=arch_name,
                                                     cut_layer=cut_layer,
                                                     base_path=base_path, exp_num=exp_num, batch_size=batch_size,
                                                     alpha_fixed=alpha_fixed,
                                                     num_clients=num_clients, bd_label=bd_label, origin_label=origin_label, tb_inj=tb_inj,
                                                     smpl_prctg=smpl_prctg, num_mlcs_cls=num_mlcs_cls)



