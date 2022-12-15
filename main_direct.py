import gc
import os
from pathlib import Path
from train_and_validation import sl, sl_simple




datasets = ['mnist']
models = ['resnet18']
num_of_exp = 1
cut_layers = [3, 4, 5, 6]
num_clients_list = [10]
num_mlcs_cls_list = [2]
tb_inj = False
alpha_list = [0.04]
save_path = '.'
tp_name = 'BASL_MARCO'
batch_size = 128
alpha_fixed = True
bd_label = 0
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
                                                     num_clients=num_clients, bd_label=bd_label, tb_inj=tb_inj,
                                                     smpl_prctg=smpl_prctg, num_mlcs_cls=num_mlcs_cls)
                            gc.collect()


