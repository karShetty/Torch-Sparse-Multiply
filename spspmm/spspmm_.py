from __future__ import annotations

import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from spspmm import _C


class _spspmm(Function):
    """
    Torch autograd Function wrapper for sparse sparse matrix (only CUDA) implementations.
    """

    @staticmethod
    def forward(
            ctx, mat1, mat2, alg_type: str = 'alg3',
    ):
        if not mat1.is_coalesced():
            mat1 = mat1.coalesce()
        if not mat2.is_coalesced():
            mat2 = mat2.coalesce()
        mat3 = _C.sparse_sparse_matmul_cuda(mat1, mat2, alg_type).coalesce()
        ctx.save_for_backward(mat1, mat2, mat3)
        ctx.alg_type = alg_type
        return mat3

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_mat3):
        mat1, mat2, mat3 = ctx.saved_tensors
        alg_type = ctx.alg_type
        grad_mat1 = _C.sparse_sparse_matmul_cuda(grad_mat3, mat2.t(), alg_type)
        grad_mat2 = _C.sparse_sparse_matmul_cuda(mat1.t(), grad_mat3, alg_type)
        return grad_mat1, grad_mat2, None,


def reshape_batch(sparse_tensor):
    # Unpack the original dimensions
    assert 3 <= len(
        sparse_tensor.shape,
    ) <= 4, 'sparse_tensor ndim must be either 3 or 4'
    assert sparse_tensor.layout == torch.sparse_coo

    # Retrieve indices and values
    indices = sparse_tensor._indices()
    values = sparse_tensor._values()

    if len(sparse_tensor.shape) == 3:
        x1, x2, x3 = sparse_tensor.shape

        # Calculate offsets for row indices based on the batch dimensions
        rows_offset = indices[0] * x2
        cols_offset = indices[0] * x3

        # Adjust indices to reflect concatenated structure
        # Adding base offset to the x2 dimension indices
        new_row_indices = rows_offset + indices[1]
        # Adding base offset to the x3 dimension indices
        new_col_indices = cols_offset + indices[2]

        # Concatenated tensor dimensions
        new_shape = (x2 * x1, x3 * x1)

    elif len(sparse_tensor.shape) == 4:
        x1, x2, x3, x4 = sparse_tensor.shape

        rows_offset = (indices[0] * x2 * x3) + (indices[1] * x3)
        cols_offset = indices[0] * x2 * x4 + indices[1] * x4

        # Adjust indices to reflect concatenated structure
        # Adding base offset to the x3 dimension indices
        new_row_indices = rows_offset + indices[2]
        # Adding base offset to the x4 dimension indices
        new_col_indices = cols_offset + indices[3]

        new_shape = (x1 * x2 * x3, x1 * x2 * x4)
    else:
        raise NotImplementedError

    # Creating new sparse tensor from adjusted indices
    new_indices = torch.stack([new_row_indices, new_col_indices])
    new_tensor = torch.sparse_coo_tensor(new_indices, values, new_shape)

    return new_tensor


def reverse_reshape_batch(reshaped_tensor, rev_shape):
    assert len(reshaped_tensor.shape) == 2, 'sparse_tensor ndim needs to be 2'
    assert 3 <= len(rev_shape) <= 4, 'sparse_tensor ndim must be either 3 or 4'
    # Extract the indices and values from the reshaped tensor
    indices = reshaped_tensor._indices()
    values = reshaped_tensor._values()

    # Calculate the original indices based on the reshaped indices
    row_indices = indices[0]
    col_indices = indices[1]

    if len(rev_shape) == 3:
        x1, x2, x3 = rev_shape

        # Reverse the calculation of new_row_indices to find original indices[0], indices[1], and indices[2]
        indices_0 = row_indices // (x2)
        remainder_0 = row_indices % (x2)
        indices_1 = remainder_0 % x2

        # Reverse the calculation of new_col_indices to find original indices[2]
        indices_2 = col_indices % x3

        # Reassemble the original indices
        original_indices = torch.stack([indices_0, indices_1, indices_2])
    elif len(rev_shape) == 4:
        x1, x2, x3, x4 = rev_shape

        # Reverse the calculation of new_row_indices to find original indices[0], indices[1], and indices[2]
        indices_0 = row_indices // (x2 * x3)
        remainder_0 = row_indices % (x2 * x3)
        indices_1 = remainder_0 // x3
        indices_2 = remainder_0 % x3

        # Reverse the calculation of new_col_indices to find original indices[3]
        indices_3 = col_indices % x4

        # Reassemble the original indices
        original_indices = torch.stack(
            [indices_0, indices_1, indices_2, indices_3],
        )
    else:
        raise NotImplementedError

    # Recreate the original tensor
    original_tensor = torch.sparse_coo_tensor(
        original_indices, values, rev_shape,
    )

    return original_tensor


def spspmm(mat1: torch.Tensor, mat2: torch.Tensor, alg_type: str = 'alg3'):
    assert mat1.layout == torch.sparse_coo
    assert mat2.layout == torch.sparse_coo
    assert mat1.device.type == 'cuda'
    assert mat2.device.type == 'cuda'
    if len(mat1.shape) == 2 and len(mat1.shape) == 2:
        return _spspmm.apply(mat1, mat2, alg_type)
    elif len(mat1.shape) == 3 and len(mat1.shape) == 3:
        b1, n, m = mat1.shape
        b2, m, k = mat2.shape
        assert b1 == b2
        mat1_2x2 = reshape_batch(mat1)
        mat2_2x2 = reshape_batch(mat2)
        mat3_2x2 = _spspmm.apply(mat1_2x2, mat2_2x2, alg_type)
        mat3_3x3 = reverse_reshape_batch(mat3_2x2, [b1, n, k])
        return mat3_3x3
    elif len(mat1.shape) == 4 and len(mat1.shape) == 4:
        b1, c1, n, m = mat1.shape
        b2, c2, m, k = mat2.shape
        assert b1 == b2
        assert c1 == c2
        mat1_2x2 = reshape_batch(mat1)
        mat2_2x2 = reshape_batch(mat2)
        mat3_2x2 = _spspmm.apply(mat1_2x2, mat2_2x2, alg_type)
        mat3_4x4 = reverse_reshape_batch(mat3_2x2, [b1, c1, n, k])
        return mat3_4x4
    else:
        raise NotImplementedError
