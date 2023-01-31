import os
import time
import torch
from torch import nn
from torch import Tensor
from typing import Tuple


def load_mnist_data(data_dir='./data') -> Tuple[Tensor, Tensor, list]:
    data_path = f"{data_dir}/FashionMNIST_data_70000x28x28.pth"
    target_path = f"{data_dir}/FashionMNIST_target_70000.pth"

    if all([os.path.isfile(path) for path in (data_path, target_path)]):
        data = torch.load(data_path)
        target = torch.load(target_path)
    else:
        from torchvision import datasets
        data_train = datasets.FashionMNIST(data_dir, train=True, download=True, )
        data_test = datasets.FashionMNIST(data_dir, train=False, download=True, )

        print(data_train)
        print(data_test)

        data = torch.cat((data_train.data, data_test.data), dim=0)
        target = torch.cat((data_train.targets, data_test.targets), dim=0)

        torch.save(data, data_path)
        torch.save(target, target_path)

    classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

    assert data.shape == (70000, 28, 28)
    assert data.dtype is torch.uint8
    assert target.shape == (70000,)
    assert target.dtype is torch.int64
    return data, target, classes


def build_mlp(dims: [int], activation: nn = None, if_raw_out: bool = True) -> nn.Sequential:
    """
    build MLP (MultiLayer Perceptron)

    dims: the middle dimension, `dims[-1]` is the output dimension of this network
    activation: the activation function
    if_remove_out_layer: if remove the activation function of the output layer.
    """
    if activation is None:
        activation = nn.ReLU
    net_list = []
    for i in range(len(dims) - 1):
        net_list.extend([nn.Linear(dims[i], dims[i + 1]), activation()])
    if if_raw_out:
        del net_list[-1]  # delete the activation function of the output layer to keep raw output
    return nn.Sequential(*net_list)


def train_mlp(if_multi_gpu: bool = False):
    gpu_id = 4
    gpu_ids = (4, 5, 6, 7)
    device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")

    '''load data'''
    data, target, classes = load_mnist_data()
    num_classes = len(classes)

    data = data.reshape(-1, 28 ** 2) / 256.
    target = torch.eye(num_classes)[target]
    data = data.to(device)
    target = target.to(device)
    assert data.shape == (70000, 28 ** 2)
    assert data.dtype is torch.float
    assert target.shape == (70000, num_classes)
    assert target.dtype is torch.float

    inp_dim = data.shape[1]
    out_dim = target.shape[1]
    mid_dim = 2 ** 8
    mlp_dims = [inp_dim, mid_dim, mid_dim, mid_dim, out_dim]

    model = build_mlp(dims=mlp_dims, if_raw_out=True).to(device)
    if if_multi_gpu:
        model = nn.DataParallel(model, device_ids=gpu_ids)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = torch.nn.SmoothL1Loss()

    from torch.nn.utils import clip_grad_norm_

    def optimizer_update():
        optimizer.zero_grad()
        obj.backward()
        clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()

    print("Train Loop:")
    num_samples = data.shape[0]
    batch_size = num_samples // 4
    repeat_times = 2 ** 5

    for i in range(int(num_samples / batch_size * repeat_times)):
        ids = torch.randint(num_samples, size=(batch_size,), requires_grad=False)
        inp = data[ids]
        lab = target[ids]

        out = model(inp)
        obj = criterion(torch.softmax(out, dim=1), lab)
        optimizer_update()

        if (i + 1) % (num_samples / batch_size) == 0:
            inp = data
            lab = target

            out = model(inp)
            obj = criterion(torch.softmax(out, dim=1), lab)
            print(f"repeat_times: {i:4}   obj: {obj.item():.3f}")
    """
;
gpu_id = 4
if_multi_gpu = False
+-------------------------------+----------------------+----------------------+
|   4  NVIDIA GeForce ...  On   | 00000000:88:00.0 Off |                  N/A |
| 13%   33C    P2   133W / 257W |   1584MiB / 11019MiB |     58%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

gpu_ids = (4, 5, 6, 7)
if_multi_gpu = True
+-------------------------------+----------------------+----------------------+
|   4  NVIDIA GeForce ...  On   | 00000000:88:00.0 Off |                  N/A |
| 13%   34C    P2    91W / 257W |   1598MiB / 11019MiB |     33%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   5  NVIDIA GeForce ...  On   | 00000000:89:00.0 Off |                  N/A |
| 13%   33C    P2    51W / 257W |   1000MiB / 11019MiB |     24%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   6  NVIDIA GeForce ...  On   | 00000000:B1:00.0 Off |                  N/A |
| 13%   33C    P2    58W / 257W |   1000MiB / 11019MiB |     24%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   7  NVIDIA GeForce ...  On   | 00000000:B2:00.0 Off |                  N/A |
| 13%   31C    P2    64W / 257W |   1000MiB / 11019MiB |     15%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
;
    """


if __name__ == '__main__':
    # load_mnist_data()
    train_mlp(if_multi_gpu=False)
    train_mlp(if_multi_gpu=True)

