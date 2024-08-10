from __future__ import annotations

import time

import torch
from torch.cuda import memory_allocated

from spspmm import spspmm

# Batch sparse multiply
a = torch.randn((4, 3, 5)).cuda()
b = torch.randn((4, 5, 3)).cuda()
result = spspmm(a.to_sparse(), b.to_sparse())


def torch_multiply(size, shape, alg_type):
    indices = torch.randint(0, shape, (2, size), device='cuda')
    values1 = torch.rand(size, device='cuda')
    values2 = torch.rand(size, device='cuda')
    matrix1 = torch.sparse_coo_tensor(
        indices, values1, (shape, shape), device='cuda',
    )
    matrix2 = torch.sparse_coo_tensor(
        indices, values2, (shape, shape), device='cuda',
    )

    spspmm(matrix1, matrix2, alg_type)  # Dummy

    start_time = time.time()
    result = spspmm(matrix1, matrix2, alg_type)
    duration = time.time() - start_time

    memory = memory_allocated('cuda')

    return duration, memory


# All matrices are of shape 20000000x20000000
for size in [100, 1000, 10000, 100000, 5000000]:
    torch_time, torch_mem = torch_multiply(size, 20000000, alg_type='alg3')
    print(f'PyTorch: Time = {torch_time}s, Memory = {torch_mem} bytes')
print('############')

for size in [100, 1000, 10000, 100000, 5000000]:
    torch_time, torch_mem = torch_multiply(size, 20000000, alg_type='alg2')
    print(f'PyTorch: Time = {torch_time}s, Memory = {torch_mem} bytes')
print('############')

for size in [100, 1000, 10000, 100000, 5000000]:
    torch_time, torch_mem = torch_multiply(size, 20000000, alg_type='alg1')
    print(f'PyTorch: Time = {torch_time}s, Memory = {torch_mem} bytes')
