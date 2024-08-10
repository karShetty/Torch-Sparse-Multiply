#pragma once
#include <torch/extension.h>

namespace at {
namespace native {

Tensor sparse_sparse_matmul_cuda12(
    const Tensor& mat1_,
    const Tensor& mat2_,
    std::string alg_name);
}
} // namespace at
