import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import Tuple
from torch.nn.parallel import DistributedDataParallel
from torch import Tensor


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

    classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

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


def run__train_in_fashion_mnist(rank_id: int = -1, world_size: int = -1):
    data_repeat: int = 64
    if_dist_data_parallel = True

    dist.init_process_group(backend='nccl', init_method='env://', rank=rank_id, world_size=world_size)
    # backend in {'nccl', 'gloo', 'mpi'}

    '''DistributedDataParallel'''
    if rank_id == -1:
        rank_id = dist.get_rank()  # the GPU_id of global_GPUs
        world_size = dist.get_world_size()  # num_global_GPUs = num_machines * num_GPU_per_machine
    num_gpus = world_size
    gpu_id = rank_id
    device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
    print(f"world_size {world_size}    rank_id {rank_id}    device {torch.zeros(1, device=device)}")

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

    '''build model'''
    inp_dim = data.shape[1]
    out_dim = target.shape[1]
    mid_dim = 2 ** 8
    mlp_dims = [inp_dim, mid_dim, mid_dim, mid_dim, out_dim]

    model = build_mlp(dims=mlp_dims, if_raw_out=True).to(device)
    if if_dist_data_parallel:
        model = DistributedDataParallel(model, device_ids=[rank_id])

    criterion = torch.nn.SmoothL1Loss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    '''dist training set'''
    dist_inp = data[rank_id::num_gpus].to(device)
    dist_lab = target[rank_id::num_gpus].to(device)

    print("Train Loop:") if rank_id == 0 else None
    num_samples = dist_inp.shape[0]
    batch_size = num_samples // 4
    repeat_times = 2 ** 4

    timer = time.time()
    for i in range(int(num_samples / batch_size * repeat_times)):
        if_evaluate = (i + 1) % (num_samples / batch_size) == 0

        ids = torch.randint(num_samples, size=(batch_size,), requires_grad=False)
        inp = dist_inp[ids]
        lab = dist_lab[ids]

        out = model(inp)
        obj = criterion(torch.softmax(out, dim=1), lab).mean()

        optimizer_update(optimizer, obj)

        if if_evaluate and rank_id == 0:
            gpu_memo = get_gpu_memo_percent(gpu_id)
            print(f"epoch: {i:4}   obj: {obj.item():.3f}    GPUMemo {gpu_memo:2.0f}")

    max_memo = get_gpu_max_memo_percent(gpu_id)
    time.sleep(rank_id * 0.1)
    print(f"TimeUsed {int(time.time() - timer):2}   MaxGPUMemo {max_memo:2.0f}")

    dist.destroy_process_group()


'''launch'''


def run_with_torch_distributed_run():
    run__train_in_fashion_mnist(rank_id=-1, world_size=-1)  # -1 means params determined by torch.distributed
    """
cd ~/workspace/DDP

https://pytorch.org/docs/stable/distributed.html#launch-utility

CUDA_VISIBLE_DEVICES="4,5,6,7" OMP_NUM_THREADS=4 \
 python -m torch.distributed.run --nproc_per_node 4 \
 DEMO_DistDataParallel.py
    """


def run_without_torch_distributed_run(gpu_ids: Tuple[int, ...] = (0, 1, 2, 3)):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_ids)[1:-1]
    os.environ['MASTER_ADDR'] = '127.0.0.1'  # 'localhost'
    os.environ['MASTER_PORT'] = '10086'

    world_size = len(gpu_ids)

    mp.set_start_method(method='spawn' if os.name == 'nt' else 'forkserver', force=True)
    processes = [mp.Process(target=run__train_in_fashion_mnist, args=(rank_id, world_size), daemon=True)
                 for rank_id in range(world_size)]
    [process.start() for process in processes]
    [process.join() for process in processes]


if __name__ == '__main__':
    # run_with_torch_distributed_run()
    run_without_torch_distributed_run(gpu_ids=(4, 5, 6, 7))

"""
data_repeat: int = 24
    world_size 4    rank_id 2
    world_size 4    rank_id 1
    world_size 4    rank_id 0
    world_size 4    rank_id 3
    Train Loop:
    repeat_times:    3   obj: 0.035    GPU_memo 15
    repeat_times:    7   obj: 0.026    GPU_memo 15
    repeat_times:   11   obj: 0.022    GPU_memo 15
    repeat_times:   15   obj: 0.020    GPU_memo 15
    repeat_times:   19   obj: 0.018    GPU_memo 15
    repeat_times:   23   obj: 0.016    GPU_memo 15
    repeat_times:   27   obj: 0.014    GPU_memo 15
    repeat_times:   31   obj: 0.014    GPU_memo 15
    TimeUsed  1   18
    TimeUsed  1   18
    TimeUsed  1   18
    TimeUsed  1   18
;

data_repeat: int = 64
    world_size 4    rank_id 2
    world_size 4    rank_id 3
    world_size 4    rank_id 1
    world_size 4    rank_id 0
    Train Loop:
    repeat_times:    3   obj: 0.049    GPUMemo 40
    repeat_times:    7   obj: 0.058    GPUMemo 40
    repeat_times:   11   obj: 0.050    GPUMemo 40
    repeat_times:   15   obj: 0.037    GPUMemo 40
    repeat_times:   19   obj: 0.033    GPUMemo 40
    repeat_times:   23   obj: 0.030    GPUMemo 40
    repeat_times:   27   obj: 0.029    GPUMemo 40
    repeat_times:   31   obj: 0.028    GPUMemo 40
    TimeUsed  2   MaxGPUMemo 47
    TimeUsed  2   MaxGPUMemo 47
    TimeUsed  2   MaxGPUMemo 47
    TimeUsed  2   MaxGPUMemo 47
;
"""
