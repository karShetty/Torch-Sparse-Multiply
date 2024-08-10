# spspmm: PyTorch Memory Efficient Sparse Sparse Matrix Multiplication


An example Pytorch module for Sparse Sparse Matrix Multiplication based on memory efficient algorithm ALG2 and ALG3.
Here is the [cusparseSpGEMM](https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuSPARSE/spgemm_mem) sample.

### Building from source

> **NOTE**: This works with only CUDA >=12.0 (tested on PyTorch 12.1).

```python
python setup.py install
```

### Running (example)

That's it! You're now ready to go. Here's a quick guide to using the package.

```python
>>> import torch
>>> from spspmm import spspmm
```

Create two random coo sparse tensors. Here, the first dim is the batch

```python
>>> a = torch.randn((4,3,5)).cuda().to_sparse()
>>> b = torch.randn((4,5,3)).cuda().to_sparse()
>>> c = spspmm(a, b)
```

Batching works based on concatenation based on [this](https://github.com/pytorch/pytorch/issues/14489#issuecomment-744523775) .
