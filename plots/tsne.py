import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# # %%
#
# (x_train, y_train), (_, _) = mnist.load_data()
# print(type(x_train))
# x_train = x_train[:3000]
# y_train = y_train[:3000]
# print(x_train.shape)
#
# print(x_train.shape)
# x_mnist = np.reshape(x_train, [x_train.shape[0], x_train.shape[1] * x_train.shape[2]])
# print(x_mnist.shape)
#
# # %%
#
# tsne = TSNE(n_components=2, verbose=1, random_state=123)
# z = tsne.fit_transform(x_mnist)
# df = pd.DataFrame()
# df["y"] = y_train
# df["comp-1"] = z[:, 0]
# df["comp-2"] = z[:, 1]
#
# sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
#                 palette=sns.color_palette("hls", 10),
#                 data=df).set(title="MNIST data T-SNE projection")
# plt.savefig('scatt.jpeg', dpi=500)


# %%

def tsne_plot(address, num_of_clients):
    smsh_dict = {}
    lbl_dict = {}
    for item in range(num_of_clients):
        load_address = address.joinpath(f'{item}.pt')
        if not load_address.exists():
            raise Exception('Such a path does not exist')
        smsh_tensor = torch.load(load_address, map_location=torch.device('cpu'))
        print(load_address)
        smsh_tensor = smsh_tensor.numpy()
        smsh_tensor = np.reshape(smsh_tensor, [smsh_tensor.shape[0], -1])
        smsh_dict[f'{item}'] = smsh_tensor
        lbl_dict[f'{item}'] = np.full(shape=smsh_tensor.shape[0], fill_value=item)
    smsh_stack = [item for item in smsh_dict.values()]
    smsh_stack = np.concatenate(smsh_stack, axis=0)
    labels_stack = [item for item in lbl_dict.values()]
    labels_stack = np.concatenate(labels_stack, axis=0)

    perplexity_list = [35, 40, 45, 50]
    n_iter_list = [1000, 2000]
    for perplexity in perplexity_list:
        for n_iter in n_iter_list:
            tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter,  verbose=1)
            z = tsne.fit_transform(smsh_stack)
            df = pd.DataFrame()
            df["label"] = labels_stack
            df["comp-1"] = z[:, 0]
            df["comp-2"] = z[:, 1]

            sns.scatterplot(x="comp-1", y="comp-2", hue='label',
                            data=df).set(title="TSNE Plot")
            plt.savefig(f'{address}/tsne_perplx{perplexity}_niter{n_iter}.jpeg', dpi=500)


def tsne_plot_per_client(smsh_address, lbl_address, num_of_clients):
    smsh_dict = {}
    lbl_dict = {}
    for item in range(num_of_clients):
        smsh_load_address = smsh_address.joinpath(f'{item}.pt')
        lbl_load_address = lbl_address.joinpath(f'{item}_lbls.pt')
        if not smsh_load_address.exists() or not lbl_load_address.exists():
            raise Exception('Such a path does not exist')
        smsh_tensor = torch.load(smsh_load_address, map_location=torch.device('cpu'))
        labels_tensor = torch.load(lbl_load_address, map_location=torch.device('cpu'))
        print(smsh_load_address, lbl_load_address)
        smsh_tensor = smsh_tensor.numpy()
        smsh_tensor = np.reshape(smsh_tensor, [smsh_tensor.shape[0], -1])
        labels_tensor = labels_tensor.numpy()


        perplexity_list = [35, 40, 45, 50]
        n_iter_list = [1000, 2000]
        for perplexity in perplexity_list:
            for n_iter in n_iter_list:
                tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter,  verbose=1)
                z = tsne.fit_transform(smsh_tensor)
                df = pd.DataFrame()
                df["label"] = labels_tensor
                df["comp-1"] = z[:, 0]
                df["comp-2"] = z[:, 1]

                sns.scatterplot(x="comp-1", y="comp-2", hue='label',
                                data=df).set(title="TSNE Plot")
                plt.savefig(f'{smsh_address}/tsne_perplx{perplexity}_niter{n_iter}.jpeg', dpi=500)


# for epoch_num in ['9', '19', '29', '39', '49', '59', '69', '79', '89', '99']:
#     for mode in ['BW', 'FW']:
#         address = f'./10clients/61/{epoch_num}/{mode}'
#         tsne_plot(address, num_of_clients=10)

for epoch_num in ['9']:
    for mode in ['BW', 'FW']:
        smsh_address = Path().joinpath('10clients', '61', f'{epoch_num}', f'{mode}')
        lbl_address = Path().joinpath('10clients', '61', f'{epoch_num}')
        if not smsh_address.exists() or not lbl_address.exists():
            raise Exception('Path does not exist')
        tsne_plot_per_client(smsh_address=smsh_address, lbl_address=lbl_address, num_of_clients=10)
