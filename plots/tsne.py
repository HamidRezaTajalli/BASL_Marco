import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from keras.datasets import mnist
from sklearn.datasets import load_iris
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# %%

(x_train, y_train), (_, _) = mnist.load_data()
print(type(x_train))
x_train = x_train[:3000]
y_train = y_train[:3000]
print(x_train.shape)

print(x_train.shape)
x_mnist = np.reshape(x_train, [x_train.shape[0], x_train.shape[1] * x_train.shape[2]])
print(x_mnist.shape)

# %%

tsne = TSNE(n_components=2, verbose=1, random_state=123)
z = tsne.fit_transform(x_mnist)
df = pd.DataFrame()
df["y"] = y_train
df["comp-1"] = z[:, 0]
df["comp-2"] = z[:, 1]

sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                palette=sns.color_palette("hls", 10),
                data=df).set(title="MNIST data T-SNE projection")
plt.savefig('scatt.jpeg', dpi=500)


# %%

def tsne_plot(address):
    smsh_dict = {}
    lbl_dict = {}
    for item in range(10):
        smsh_tensor = torch.load(address+f'/{item}.pt', map_location=torch.device('cpu'))
        smsh_tensor = smsh_tensor.numpy()
        smsh_tensor = np.reshape(smsh_tensor, [smsh_tensor.shape[0], -1])
        smsh_dict[f'{item}'] = smsh_tensor
        lbl_dict[f'{item}'] = np.full(shape=smsh_tensor.shape[0], fill_value=item)
    smsh_stack = [item for item in smsh_dict.values()]
    smsh_stack = np.concatenate(smsh_stack, axis=0)
    labels_stack = [item for item in lbl_dict.values()]
    labels_stack = np.concatenate(labels_stack, axis=0)

    tsne = TSNE(n_components=2, verbose=1)
    z = tsne.fit_transform(smsh_stack)
    df = pd.DataFrame()
    df["y"] = labels_stack
    df["comp-1"] = z[:, 0]
    df["comp-2"] = z[:, 1]

    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                    palette=sns.color_palette("hls", 10),
                    data=df).set(title="First Try")
    plt.savefig('scatt.jpeg', dpi=500)


for epoch_num in ['9', '19', '29', '39', '49', '59', '69', '79', '89', '99']:
    for mode in ['BW', 'FW']:
        address = f'./10clients/61/{epoch_num}/{mode}'
        tsne_plot(address)
