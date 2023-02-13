import os
import sys
import torch
from time import time
from torch import nn
from torch import Tensor
from typing import Tuple


def load_mnist_data(data_dir='./data') -> Tuple[Tensor, Tensor, list]:
    data_path = f"{data_dir}/FashionMNIST_data_70000x28x28.pth"
    target_path = f"{data_dir}/FashionMNIST_target_70000.pth"

    if all([os.path.isfile(path) for path in (data_path, target_path)]):
        data = torch.load(data_path, map_location=torch.device('cpu'))
        target = torch.load(target_path, map_location=torch.device('cpu'))
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


class FullModel(nn.Module):
    def __init__(self, model, loss):
        super().__init__()
        self.model = model
        self.loss = loss

    def forward(self, inputs, targets):
        outputs = self.model(inputs)
        return self.loss(outputs, targets)


def get_gpu_memo_percent(gpu_id: int) -> float:
    return (torch.cuda.memory_allocated(gpu_id)
            / torch.cuda.get_device_properties(gpu_id).total_memory) * 99.


def get_gpu_max_memo_percent(gpu_id: int) -> float:
    return (torch.cuda.max_memory_allocated(gpu_id)
            / torch.cuda.get_device_properties(gpu_id).total_memory) * 99.


def optimizer_update(optimizer, objective):
    optimizer.zero_grad()
    objective.backward()
    optimizer.step()


def train_mlp_dp():
    data_repeat: int = 24
    if_full_model = bool(int(sys.argv[1]))
    if_data_parallel = bool(int(sys.argv[2]))
    if_save_data_in_gpu = bool(int(sys.argv[3]))

    gpu_ids = (4, 5, 6, 7)
    gpu_id = gpu_ids[0]

    device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")

    '''load data'''
    data, target, classes = load_mnist_data()
    num_classes = len(classes)

    data = data.reshape(-1, 28 ** 2) / 256.
    target = torch.eye(num_classes)[target]
    assert data.shape == (70000, 28 ** 2)
    assert data.dtype is torch.float
    assert target.shape == (70000, num_classes)
    assert target.dtype is torch.float

    if data_repeat > 1:
        data = data.repeat(1, data_repeat)

    inp_dim = data.shape[1]
    out_dim = target.shape[1]
    mid_dim = 2 ** 8
    mlp_dims = [inp_dim, mid_dim, mid_dim, mid_dim, out_dim]

    model = build_mlp(dims=mlp_dims, if_raw_out=True)
    criterion = torch.nn.SmoothL1Loss(reduction='none')

    if if_full_model:
        model = FullModel(model=model, loss=criterion)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    if if_data_parallel:
        model = nn.DataParallel(model, device_ids=gpu_ids)

    if if_save_data_in_gpu:
        data = data.to(device)
        target = target.to(device)
    model = model.to(device)

    print("Train Loop:")
    num_samples = data.shape[0]
    batch_size = num_samples // 4
    repeat_times = 2 ** 3

    timer = time()
    for i in range(int(num_samples / batch_size * repeat_times)):
        if_evaluate = (i + 1) % (num_samples / batch_size) == 0

        ids = torch.randint(num_samples, size=(batch_size,), requires_grad=False)
        inp = data[ids]
        lab = target[ids]

        if (not if_save_data_in_gpu) and (not if_data_parallel):
            inp = inp.to(device)
            lab = lab.to(device)

        if if_full_model:
            obj = model(inp, lab).mean()
        else:
            out = model(inp)
            obj = criterion(torch.softmax(out, dim=1), lab).mean()

        optimizer_update(optimizer, obj)

        if if_evaluate:
            memory_memo0 = get_gpu_memo_percent(gpu_id)
            memory_memo_ = sum([get_gpu_memo_percent(i) for i in gpu_ids[1:]]) / len(gpu_ids[1:])
            print(f"epoch: {i:4}   obj: {obj.item():.3f}    GPU_memo {memory_memo0:2.0f} {memory_memo_:2.0f}")

    max_memory_memo0 = get_gpu_max_memo_percent(gpu_id)
    max_memory_memo_ = max([get_gpu_max_memo_percent(i) for i in gpu_ids[1:]])
    print(f"TimeUsed {int(time() - timer):2}    GPU_max_memo {max_memory_memo0:2.0f} {max_memory_memo_:2.0f}")


"""
if_full_model
    if_data_parallel 
        if_save_data_in_gpu
                                        主线程的GPU显存占用
                                            子进程的GPU显存占用

实验：是否开启DataParallel 给并行GPU卡的影响
0   0   1   TimeUsed  1    GPU_max_memo 68  0
0   1   1   TimeUsed 12    GPU_max_memo 68  3

问题：并行GPU中，一张GPU显存占用高，其他占用低的问题，有方法能降低最高显存占用吗？
结论：不把数据放在GPU内存里，的确能减小显存占用，但是会增加训练时间
实验：不把数据放在GPU内存里
0   1   0   ERROR 将数据放在CPU上，不开启 full_model 就会报错
1   1   0   TimeUsed 19    GPU_max_memo  4  3
1   0   0   TimeUsed 13    GPU_max_memo 13  0


问题：full_model 不得不用，那么它是否会影响训练？
结论：是否开启full_model，对训练时间，显存占用影响小到看不见
实验：是否开启full_model 在单卡下的影响
0   0   1   TimeUsed  1    GPU_max_memo 68  0
1   0   1   TimeUsed  1    GPU_max_memo 68  0
实验：是否开启full_model 在多卡下的影响
0   1   1   TimeUsed 12    GPU_max_memo 68  3
1   1   1   TimeUsed 12    GPU_max_memo 68  3
"""


def train_mlp_ddp():
    data_repeat: int = 24
    if_full_model = bool(int(sys.argv[1]))
    if_data_parallel = bool(int(sys.argv[2]))
    if_save_data_in_gpu = bool(int(sys.argv[3]))

    gpu_ids = (4, 5, 6, 7)
    gpu_id = gpu_ids[0]

    device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")

    '''load data'''
    data, target, classes = load_mnist_data()
    num_classes = len(classes)

    data = data.reshape(-1, 28 ** 2) / 256.
    target = torch.eye(num_classes)[target]
    assert data.shape == (70000, 28 ** 2)
    assert data.dtype is torch.float
    assert target.shape == (70000, num_classes)
    assert target.dtype is torch.float

    if data_repeat > 1:
        data = data.repeat(1, data_repeat)

    inp_dim = data.shape[1]
    out_dim = target.shape[1]
    mid_dim = 2 ** 8
    mlp_dims = [inp_dim, mid_dim, mid_dim, mid_dim, out_dim]

    model = build_mlp(dims=mlp_dims, if_raw_out=True)
    criterion = torch.nn.SmoothL1Loss(reduction='none')

    if if_full_model:
        model = FullModel(model=model, loss=criterion)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    if if_data_parallel:
        model = nn.DataParallel(model, device_ids=gpu_ids)

    if if_save_data_in_gpu:
        data = data.to(device)
        target = target.to(device)
    model = model.to(device)

    print("Train Loop:")
    num_samples = data.shape[0]
    batch_size = num_samples // 4
    repeat_times = 2 ** 3

    timer = time()
    for i in range(int(num_samples / batch_size * repeat_times)):
        if_evaluate = (i + 1) % (num_samples / batch_size) == 0

        ids = torch.randint(num_samples, size=(batch_size,), requires_grad=False)
        inp = data[ids]
        lab = target[ids]

        if (not if_save_data_in_gpu) and (not if_data_parallel):
            inp = inp.to(device)
            lab = lab.to(device)

        if if_full_model:
            obj = model(inp, lab).mean()
        else:
            out = model(inp)
            obj = criterion(torch.softmax(out, dim=1), lab).mean()

        optimizer_update(optimizer, obj)

        if if_evaluate:
            memory_memo0 = get_gpu_memo_percent(gpu_id)
            memory_memo_ = sum([get_gpu_memo_percent(i) for i in gpu_ids[1:]]) / len(gpu_ids[1:])
            print(f"repeat_times: {i:4}   obj: {obj.item():.3f}    GPU_memo {memory_memo0:2.0f} {memory_memo_:2.0f}")

    max_memory_memo0 = get_gpu_max_memo_percent(gpu_id)
    max_memory_memo_ = max([get_gpu_max_memo_percent(i) for i in gpu_ids[1:]])
    print(f"TimeUsed {int(time() - timer):2}    GPU_max_memo {max_memory_memo0:2.0f} {max_memory_memo_:2.0f}")


if __name__ == '__main__':
    # train_mlp_dp()
    train_mlp_ddp()
