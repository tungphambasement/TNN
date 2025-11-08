#include "cuda/error_handler.hpp"
#include "cuda/helpers.hpp"

#include <cuda_runtime.h>

namespace cuda {
void synchronize() { cudaDeviceSynchronize(); }
} // namespace cuda