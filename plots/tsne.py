import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
import numpy as np
import seaborn as sns
import pandas as pd
from pathlib import Path
import argparse
import time

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

def tsne_plot(expname, address, num_of_clients, plt_mode: str):
    smsh_dict = {}
    lbl_dict = {}
    for item in range(num_of_clients):
        load_address = address.joinpath(f'{item}.pt')
        if not load_address.exists():
            raise Exception('Such a path does not exist')
        smsh_tensor = torch.load(load_address, map_location=torch.device('cpu'))
        print(load_address)
        smsh_tensor = smsh_tensor.detach().numpy()
        smsh_tensor = np.reshape(smsh_tensor, [smsh_tensor.shape[0], -1])
        smsh_dict[f'{item}'] = smsh_tensor
        lbl_dict[f'{item}'] = np.full(shape=smsh_tensor.shape[0], fill_value=item)
    smsh_stack = [item for item in smsh_dict.values()]
    smsh_stack = np.concatenate(smsh_stack, axis=0)
    labels_stack = [item for item in lbl_dict.values()]
    labels_stack = np.concatenate(labels_stack, axis=0)
    # print(np.unique(labels_stack, return_counts=True))

    perplexity_list = [25, 30, 35, 40, 45]
    n_iter_list = [1000]
    if plt_mode.lower() == '2d':
        for perplexity in perplexity_list:
            for n_iter in n_iter_list:
                tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, verbose=1)
                z = tsne.fit_transform(smsh_stack)
                df = pd.DataFrame()
                df["label"] = labels_stack
                df["comp-1"] = z[:, 0]
                df["comp-2"] = z[:, 1]

                sns.scatterplot(x="comp-1", y="comp-2", hue='label',
                                data=df, palette=sns.color_palette("Paired", num_of_clients), legend="brief").set(
                    title="TSNE Plot")
                plt.savefig(f'{address}/{plt_mode}_tsne_perplx{perplexity}_niter{n_iter}_{expname}.jpeg', dpi=500)
    elif plt_mode.lower() == '3d':
        for perplexity in perplexity_list:
            for n_iter in n_iter_list:
                tsne = TSNE(n_components=3, perplexity=perplexity, n_iter=n_iter, verbose=1)
                z = tsne.fit_transform(smsh_stack)
                df = pd.DataFrame()
                df["label"] = labels_stack
                df["comp-1"] = z[:, 0]
                df["comp-2"] = z[:, 1]
                df["comp-3"] = z[:, 2]

                # axes instance
                fig = plt.figure(figsize=(15, 10))
                ax = Axes3D(fig, auto_add_to_figure=False)
                fig.add_axes(ax)

                # get colormap from seaborn
                cmap = ListedColormap(sns.color_palette("Paired", num_of_clients).as_hex())

                # plot
                sc = ax.scatter3D(xs="comp-1", ys="comp-2", zs="comp-3", s=40, c="label", data=df, marker='o',
                                  cmap=cmap,
                                  edgecolors='white', alpha=1)
                # ax.set_xlabel('X Label')
                # ax.set_ylabel('Y Label')
                # ax.set_zlabel('Z Label')

                # legend
                plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)

                # save
                plt.savefig(f'{address}/{plt_mode}_tsne_perplx{perplexity}_niter{n_iter}_{expname}.jpeg', dpi=500)
    else:
        raise Exception(f"please insert correct required dimension: 2d or 3d. you have inserted {plt_mode}")


def tsne_plot_per_client(expname, smsh_address, lbl_address, num_of_clients, plt_mode: str):
    for item in range(num_of_clients):
        smsh_load_address = smsh_address.joinpath(f'{item}.pt')
        lbl_load_address = lbl_address.joinpath(f'{item}_lbls.pt')
        if not smsh_load_address.exists() or not lbl_load_address.exists():
            raise Exception('Such a path does not exist')
        smsh_tensor = torch.load(smsh_load_address, map_location=torch.device('cpu'))
        labels_tensor = torch.load(lbl_load_address, map_location=torch.device('cpu'))
        print(smsh_load_address, lbl_load_address)
        smsh_tensor = smsh_tensor.detach().numpy()
        smsh_tensor = np.reshape(smsh_tensor, [smsh_tensor.shape[0], -1])
        labels_tensor = labels_tensor.detach().numpy()

        perplexity_list = [25, 30, 35, 40, 45]
        n_iter_list = [1000]
        if plt_mode.lower() == '2d':
            for perplexity in perplexity_list:
                for n_iter in n_iter_list:
                    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, verbose=1)
                    z = tsne.fit_transform(smsh_tensor)
                    df = pd.DataFrame()
                    df["label"] = labels_tensor
                    df["comp-1"] = z[:, 0]
                    df["comp-2"] = z[:, 1]

                    sns.scatterplot(x="comp-1", y="comp-2", hue='label',
                                    data=df, palette=sns.color_palette("Paired", len(np.unique(labels_tensor))),
                                    legend="brief").set(title="TSNE Plot")
                    plt.savefig(
                        f'{smsh_address}/{plt_mode}_tsne_perclient_perplx{perplexity}_niter{n_iter}_{expname}.jpeg',
                        dpi=500)
        elif plt_mode.lower() == '3d':
            for perplexity in perplexity_list:
                for n_iter in n_iter_list:
                    tsne = TSNE(n_components=3, perplexity=perplexity, n_iter=n_iter, verbose=1)
                    z = tsne.fit_transform(smsh_tensor)
                    df = pd.DataFrame()
                    df["label"] = labels_tensor
                    df["comp-1"] = z[:, 0]
                    df["comp-2"] = z[:, 1]
                    df["comp-3"] = z[:, 2]

                    # axes instance
                    fig = plt.figure(figsize=(15, 10))
                    ax = Axes3D(fig, auto_add_to_figure=False)
                    fig.add_axes(ax)

                    # get colormap from seaborn
                    cmap = ListedColormap(sns.color_palette("Paired", len(np.unique(labels_tensor))).as_hex())

                    # plot
                    sc = ax.scatter3D(xs="comp-1", ys="comp-2", zs="comp-3", s=40, c="label", data=df, marker='o',
                                      cmap=cmap,
                                      edgecolors='white', alpha=1)
                    # ax.set_xlabel('X Label')
                    # ax.set_ylabel('Y Label')
                    # ax.set_zlabel('Z Label')

                    # legend
                    plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)

                    # save
                    plt.savefig(
                        f'{smsh_address}/{plt_mode}_tsne_perclient_perplx{perplexity}_niter{n_iter}_{expname}.jpeg',
                        dpi=500)
        else:
            raise Exception(f"please insert correct required dimension: 2d or 3d. you have inserted {plt_mode}")


# for epoch_num in ['9', '19', '29', '39', '49', '59', '69', '79', '89', '99']:
#     for mode in ['BW', 'FW']:
#         address = f'./10clients/61/{epoch_num}/{mode}'
#         tsne_plot(address, num_of_clients=10)


