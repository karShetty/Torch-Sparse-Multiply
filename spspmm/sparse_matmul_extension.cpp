#include <c10/core/Scalar.h>
#include <torch/extension.h>
#include "sparse.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "sparse_sparse_matmul_cuda",
      &at::native::sparse_sparse_matmul_cuda12,
      "Perform sparse matrix multiplication");
}
