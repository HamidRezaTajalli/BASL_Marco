import os

datasets = ['mnist', 'cifar10', 'fmnist']
models = ['resnet9', 'lenet', 'stripnet']
num_of_exp = 1
cut_layers = [1, 2, 3]
num_clients_list = [1, 3, 5, 7]
fixed_alpha = True
tb_inj = False
alpha_list = [0.5, 0.2, 0.09, 0.06, 0.04]
save_path = '.'

for dataset in datasets:
    for model in models:
        for num_clients in num_clients_list:
            for cut_layer in cut_layers:
                for exp_num in range(num_of_exp):
                    for alpha in alpha_list:
                        command = f"python3 main.py --dataname {dataset} --model {model} --exp_num {exp_num} --cutlayer {cut_layer} --num_clients {num_clients} --fixed_alpha --alpha {alpha}"
                        os.system(command)
                    # !python3 main.py --dataname $dataset --model $model --exp_num $exp_num --cutlayer $cut_layer --num_clients $num_clients --fixed_alpha --alpha $alpha
