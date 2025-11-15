#include "ops/cuda/kernels.hpp"

#ifdef USE_CUDA

#include "cuda/error_handler.hpp" // Assuming this contains checkCudaError

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>

namespace tnn {
namespace cuda {

// Kernel execution configuration helper
constexpr int BLOCK_SIZE = 256;

inline int get_num_blocks(size_t size) { return (size + BLOCK_SIZE - 1) / BLOCK_SIZE; }

// --- Kernel Definitions (No change needed here) ---

// Basic arithmetic kernels for float
__global__ void add_kernel(const float *a, const float *b, float *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = a[idx] + b[idx];
  }
}

// ... (Other float/double arithmetic kernels remain the same) ...
__global__ void sub_kernel(const float *a, const float *b, float *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = a[idx] - b[idx];
  }
}

__global__ void mul_kernel(const float *a, const float *b, float *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = a[idx] * b[idx];
  }
}

__global__ void div_kernel(const float *a, const float *b, float *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = a[idx] / b[idx];
  }
}

__global__ void add_kernel(const double *a, const double *b, double *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = a[idx] + b[idx];
  }
}

__global__ void sub_kernel(const double *a, const double *b, double *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = a[idx] - b[idx];
  }
}

__global__ void mul_kernel(const double *a, const double *b, double *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = a[idx] * b[idx];
  }
}

__global__ void div_kernel(const double *a, const double *b, double *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = a[idx] / b[idx];
  }
}

// Fused multiply-add kernels for float
__global__ void fmadd_kernel(const float *a, const float *b, float *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = fmaf(a[idx], b[idx], c[idx]);
  }
}

__global__ void fmsub_kernel(const float *a, const float *b, float *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = fmaf(a[idx], b[idx], -c[idx]);
  }
}

__global__ void fnmadd_kernel(const float *a, const float *b, float *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = fmaf(-a[idx], b[idx], c[idx]);
  }
}

// Fused multiply-add kernels for double
__global__ void fmadd_kernel(const double *a, const double *b, double *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = fma(a[idx], b[idx], c[idx]);
  }
}

__global__ void fmsub_kernel(const double *a, const double *b, double *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = fma(a[idx], b[idx], -c[idx]);
  }
}

__global__ void fnmadd_kernel(const double *a, const double *b, double *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = fma(-a[idx], b[idx], c[idx]);
  }
}

// Scalar operation kernels for float
__global__ void add_scalar_kernel(const float *a, float scalar, float *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = a[idx] + scalar;
  }
}

__global__ void mul_scalar_kernel(const float *a, float scalar, float *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = a[idx] * scalar;
  }
}

__global__ void div_scalar_kernel(const float *a, float scalar, float *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = a[idx] / scalar;
  }
}

__global__ void set_scalar_kernel(float *c, float scalar, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = scalar;
  }
}

// Scalar operation kernels for double
__global__ void add_scalar_kernel(const double *a, double scalar, double *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = a[idx] + scalar;
  }
}

__global__ void mul_scalar_kernel(const double *a, double scalar, double *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = a[idx] * scalar;
  }
}

__global__ void div_scalar_kernel(const double *a, double scalar, double *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = a[idx] / scalar;
  }
}

__global__ void set_scalar_kernel(double *c, double scalar, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = scalar;
  }
}

// Mathematical function kernels for float
__global__ void sqrt_kernel(const float *a, float *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = sqrtf(a[idx]);
  }
}

__global__ void rsqrt_kernel(const float *a, float *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = rsqrtf(a[idx]);
  }
}

__global__ void rcp_kernel(const float *a, float *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = 1.0f / a[idx];
  }
}

__global__ void abs_kernel(const float *a, float *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = fabsf(a[idx]);
  }
}

// Mathematical function kernels for double
__global__ void sqrt_kernel(const double *a, double *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = sqrt(a[idx]);
  }
}

__global__ void abs_kernel(const double *a, double *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = fabs(a[idx]);
  }
}

// Min/Max operation kernels for float
__global__ void min_kernel(const float *a, const float *b, float *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = fminf(a[idx], b[idx]);
  }
}

__global__ void max_kernel(const float *a, const float *b, float *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = fmaxf(a[idx], b[idx]);
  }
}

__global__ void scalar_max_kernel(const float *a, float scalar, float *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = fmaxf(a[idx], scalar);
  }
}

__global__ void clamp_kernel(const float *a, float min_val, float max_val, float *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = fmaxf(min_val, fminf(max_val, a[idx]));
  }
}

// Min/Max operation kernels for double
__global__ void min_kernel(const double *a, const double *b, double *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = fmin(a[idx], b[idx]);
  }
}

__global__ void max_kernel(const double *a, const double *b, double *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = fmax(a[idx], b[idx]);
  }
}

__global__ void scalar_max_kernel(const double *a, double scalar, double *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = fmax(a[idx], scalar);
  }
}

__global__ void clamp_kernel(const double *a, double min_val, double max_val, double *c,
                             size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = fmax(min_val, fmin(max_val, a[idx]));
  }
}

// Comparison operation kernels for float
__global__ void equal_kernel(const float *a, const float *b, float *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = (a[idx] == b[idx]) ? 1.0f : 0.0f;
  }
}

__global__ void greater_kernel(const float *a, const float *b, float *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = (a[idx] > b[idx]) ? 1.0f : 0.0f;
  }
}

// Comparison operation kernels for double
__global__ void equal_kernel(const double *a, const double *b, double *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = (a[idx] == b[idx]) ? 1.0 : 0.0;
  }
}

__global__ void greater_kernel(const double *a, const double *b, double *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = (a[idx] > b[idx]) ? 1.0 : 0.0;
  }
}

// Memory operation kernels for float
__global__ void copy_kernel(const float *a, float *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = a[idx];
  }
}

__global__ void zero_kernel(float *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = 0.0f;
  }
}

// Memory operation kernels for double
__global__ void copy_kernel(const double *a, double *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = a[idx];
  }
}

__global__ void zero_kernel(double *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = 0.0;
  }
}

// Specialized BatchNorm operation kernels for float
__global__ void sub_mul_scalar_kernel(const float *a, float sub_scalar, float mul_scalar, float *c,
                                      size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = (a[idx] - sub_scalar) * mul_scalar;
  }
}

__global__ void mul_add_scalar_kernel(const float *a, float mul_scalar, float add_scalar, float *c,
                                      size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = a[idx] * mul_scalar + add_scalar;
  }
}

// Specialized BatchNorm operation kernels for double
__global__ void sub_mul_scalar_kernel(const double *a, double sub_scalar, double mul_scalar,
                                      double *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = (a[idx] - sub_scalar) * mul_scalar;
  }
}

__global__ void mul_add_scalar_kernel(const double *a, double mul_scalar, double add_scalar,
                                      double *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = a[idx] * mul_scalar + add_scalar;
  }
}

// Reduction operation kernels (remain the same)

__global__ void sum_kernel(const float *a, float *result, size_t size) {
  extern __shared__ float sdata[];

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  sdata[tid] = (idx < size) ? a[idx] : 0.0f;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    result[blockIdx.x] = sdata[0];
  }
}

__global__ void dot_product_kernel(const float *a, const float *b, float *result, size_t size) {
  extern __shared__ float sdata[];

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  sdata[tid] = (idx < size) ? a[idx] * b[idx] : 0.0f;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    result[blockIdx.x] = sdata[0];
  }
}

__global__ void sum_squared_diff_kernel(const float *a, float mean, float *result, size_t size) {
  extern __shared__ float sdata[];

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size) {
    float diff = a[idx] - mean;
    sdata[tid] = diff * diff;
  } else {
    sdata[tid] = 0.0f;
  }
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    result[blockIdx.x] = sdata[0];
  }
}

__global__ void sum_kernel(const double *a, double *result, size_t size) {
  extern __shared__ double sdata_double[];

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  sdata_double[tid] = (idx < size) ? a[idx] : 0.0;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata_double[tid] += sdata_double[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    result[blockIdx.x] = sdata_double[0];
  }
}

__global__ void dot_product_kernel(const double *a, const double *b, double *result, size_t size) {
  extern __shared__ double sdata_double[];

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  sdata_double[tid] = (idx < size) ? a[idx] * b[idx] : 0.0;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata_double[tid] += sdata_double[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    result[blockIdx.x] = sdata_double[0];
  }
}

__global__ void sum_squared_diff_kernel(const double *a, double mean, double *result, size_t size) {
  extern __shared__ double sdata_double[];

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size) {
    double diff = a[idx] - mean;
    sdata_double[tid] = diff * diff;
  } else {
    sdata_double[tid] = 0.0;
  }
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata_double[tid] += sdata_double[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    result[blockIdx.x] = sdata_double[0];
  }
}

// --- Host Wrapper Functions (All non-reduction functions now take cudaStream_t) ---

// Host wrapper functions - Basic arithmetic operations for float
void cuda_add(const float *a, const float *b, float *c, size_t size, cudaStream_t stream) {
  int num_blocks = get_num_blocks(size);
  add_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(a, b, c, size);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

void cuda_sub(const float *a, const float *b, float *c, size_t size, cudaStream_t stream) {
  int num_blocks = get_num_blocks(size);
  sub_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(a, b, c, size);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

void cuda_mul(const float *a, const float *b, float *c, size_t size, cudaStream_t stream) {
  int num_blocks = get_num_blocks(size);
  mul_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(a, b, c, size);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

void cuda_div(const float *a, const float *b, float *c, size_t size, cudaStream_t stream) {
  int num_blocks = get_num_blocks(size);
  div_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(a, b, c, size);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

// Host wrapper functions - Basic arithmetic operations for double
void cuda_add(const double *a, const double *b, double *c, size_t size, cudaStream_t stream) {
  int num_blocks = get_num_blocks(size);
  add_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(a, b, c, size);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

void cuda_sub(const double *a, const double *b, double *c, size_t size, cudaStream_t stream) {
  int num_blocks = get_num_blocks(size);
  sub_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(a, b, c, size);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

void cuda_mul(const double *a, const double *b, double *c, size_t size, cudaStream_t stream) {
  int num_blocks = get_num_blocks(size);
  mul_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(a, b, c, size);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

void cuda_div(const double *a, const double *b, double *c, size_t size, cudaStream_t stream) {
  int num_blocks = get_num_blocks(size);
  div_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(a, b, c, size);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

// Host wrapper functions - Fused multiply-add operations for float
void cuda_fmadd(const float *a, const float *b, float *c, size_t size, cudaStream_t stream) {
  int num_blocks = get_num_blocks(size);
  fmadd_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(a, b, c, size);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

void cuda_fmsub(const float *a, const float *b, float *c, size_t size, cudaStream_t stream) {
  int num_blocks = get_num_blocks(size);
  fmsub_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(a, b, c, size);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

void cuda_fnmadd(const float *a, const float *b, float *c, size_t size, cudaStream_t stream) {
  int num_blocks = get_num_blocks(size);
  fnmadd_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(a, b, c, size);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

// Host wrapper functions - Fused multiply-add operations for double
void cuda_fmadd(const double *a, const double *b, double *c, size_t size, cudaStream_t stream) {
  int num_blocks = get_num_blocks(size);
  fmadd_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(a, b, c, size);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

void cuda_fmsub(const double *a, const double *b, double *c, size_t size, cudaStream_t stream) {
  int num_blocks = get_num_blocks(size);
  fmsub_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(a, b, c, size);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

void cuda_fnmadd(const double *a, const double *b, double *c, size_t size, cudaStream_t stream) {
  int num_blocks = get_num_blocks(size);
  fnmadd_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(a, b, c, size);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

// Host wrapper functions - Scalar operations for float
void cuda_add_scalar(const float *a, float scalar, float *c, size_t size, cudaStream_t stream) {
  int num_blocks = get_num_blocks(size);
  add_scalar_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(a, scalar, c, size);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

void cuda_mul_scalar(const float *a, float scalar, float *c, size_t size, cudaStream_t stream) {
  int num_blocks = get_num_blocks(size);
  mul_scalar_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(a, scalar, c, size);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

void cuda_div_scalar(const float *a, float scalar, float *c, size_t size, cudaStream_t stream) {
  int num_blocks = get_num_blocks(size);
  div_scalar_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(a, scalar, c, size);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

void cuda_set_scalar(float *c, float scalar, size_t size, cudaStream_t stream) {
  // Using cudaMemsetAsync is generally preferred for zeroing/setting constants on the GPU.
  // Note: cudaMemsetAsync only supports char (8-bit) values, so setting a non-zero float/double
  // requires a custom kernel or a library call like cublas/thrust, or a kernel launch like below.
  int num_blocks = get_num_blocks(size);
  set_scalar_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(c, scalar, size);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

// Host wrapper functions - Scalar operations for double
void cuda_add_scalar(const double *a, double scalar, double *c, size_t size, cudaStream_t stream) {
  int num_blocks = get_num_blocks(size);
  add_scalar_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(a, scalar, c, size);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

void cuda_mul_scalar(const double *a, double scalar, double *c, size_t size, cudaStream_t stream) {
  int num_blocks = get_num_blocks(size);
  mul_scalar_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(a, scalar, c, size);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

void cuda_div_scalar(const double *a, double scalar, double *c, size_t size, cudaStream_t stream) {
  int num_blocks = get_num_blocks(size);
  div_scalar_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(a, scalar, c, size);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

void cuda_set_scalar(double *c, double scalar, size_t size, cudaStream_t stream) {
  int num_blocks = get_num_blocks(size);
  set_scalar_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(c, scalar, size);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

// Host wrapper functions - Mathematical functions for float
void cuda_sqrt(const float *a, float *c, size_t size, cudaStream_t stream) {
  int num_blocks = get_num_blocks(size);
  sqrt_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(a, c, size);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

void cuda_rsqrt(const float *a, float *c, size_t size, cudaStream_t stream) {
  int num_blocks = get_num_blocks(size);
  rsqrt_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(a, c, size);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

void cuda_rcp(const float *a, float *c, size_t size, cudaStream_t stream) {
  int num_blocks = get_num_blocks(size);
  rcp_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(a, c, size);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

void cuda_abs(const float *a, float *c, size_t size, cudaStream_t stream) {
  int num_blocks = get_num_blocks(size);
  abs_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(a, c, size);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

// Host wrapper functions - Mathematical functions for double
void cuda_sqrt(const double *a, double *c, size_t size, cudaStream_t stream) {
  int num_blocks = get_num_blocks(size);
  sqrt_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(a, c, size);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

void cuda_abs(const double *a, double *c, size_t size, cudaStream_t stream) {
  int num_blocks = get_num_blocks(size);
  abs_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(a, c, size);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

// Host wrapper functions - Min/Max operations for float
void cuda_min(const float *a, const float *b, float *c, size_t size, cudaStream_t stream) {
  int num_blocks = get_num_blocks(size);
  min_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(a, b, c, size);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

void cuda_max(const float *a, const float *b, float *c, size_t size, cudaStream_t stream) {
  int num_blocks = get_num_blocks(size);
  max_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(a, b, c, size);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

void cuda_scalar_max(const float *a, float scalar, float *c, size_t size, cudaStream_t stream) {
  int num_blocks = get_num_blocks(size);
  scalar_max_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(a, scalar, c, size);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

void cuda_clamp(const float *a, float min_val, float max_val, float *c, size_t size,
                cudaStream_t stream) {
  int num_blocks = get_num_blocks(size);
  clamp_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(a, min_val, max_val, c, size);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

// Host wrapper functions - Min/Max operations for double
void cuda_min(const double *a, const double *b, double *c, size_t size, cudaStream_t stream) {
  int num_blocks = get_num_blocks(size);
  min_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(a, b, c, size);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

void cuda_max(const double *a, const double *b, double *c, size_t size, cudaStream_t stream) {
  int num_blocks = get_num_blocks(size);
  max_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(a, b, c, size);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

void cuda_scalar_max(const double *a, double scalar, double *c, size_t size, cudaStream_t stream) {
  int num_blocks = get_num_blocks(size);
  scalar_max_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(a, scalar, c, size);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

void cuda_clamp(const double *a, double min_val, double max_val, double *c, size_t size,
                cudaStream_t stream) {
  int num_blocks = get_num_blocks(size);
  clamp_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(a, min_val, max_val, c, size);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

// Host wrapper functions - Comparison operations for float
void cuda_equal(const float *a, const float *b, float *c, size_t size, cudaStream_t stream) {
  int num_blocks = get_num_blocks(size);
  equal_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(a, b, c, size);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

void cuda_greater(const float *a, const float *b, float *c, size_t size, cudaStream_t stream) {
  int num_blocks = get_num_blocks(size);
  greater_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(a, b, c, size);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

// Host wrapper functions - Comparison operations for double
void cuda_equal(const double *a, const double *b, double *c, size_t size, cudaStream_t stream) {
  int num_blocks = get_num_blocks(size);
  equal_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(a, b, c, size);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

void cuda_greater(const double *a, const double *b, double *c, size_t size, cudaStream_t stream) {
  int num_blocks = get_num_blocks(size);
  greater_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(a, b, c, size);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

// Host wrapper functions - Memory operations for float
void cuda_copy(const float *a, float *c, size_t size, cudaStream_t stream) {
  // Use cudaMemcpyAsync for asynchronous copy
  cudaMemcpyAsync(c, a, size * sizeof(float), cudaMemcpyDeviceToDevice, stream);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

void cuda_zero(float *c, size_t size, cudaStream_t stream) {
  // Use cudaMemsetAsync for asynchronous zeroing (only works for 8-bit value 0)
  cudaMemsetAsync(c, 0, size * sizeof(float), stream);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

// Host wrapper functions - Memory operations for double
void cuda_copy(const double *a, double *c, size_t size, cudaStream_t stream) {
  cudaMemcpyAsync(c, a, size * sizeof(double), cudaMemcpyDeviceToDevice, stream);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

void cuda_zero(double *c, size_t size, cudaStream_t stream) {
  cudaMemsetAsync(c, 0, size * sizeof(double), stream);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

// Host wrapper functions - Reduction operations for float (These remain SYNCHRONOUS)
float cuda_sum(const float *a, size_t size, cudaStream_t stream) {
  int num_blocks = get_num_blocks(size);

  // Allocate device memory for intermediate results
  float *d_temp_result;
  cudaMalloc(&d_temp_result, num_blocks * sizeof(float));

  // Launch kernel (synchronous launch on default stream)
  sum_kernel<<<num_blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float), stream>>>(a, d_temp_result,
                                                                             size);

  // If we have multiple blocks, we need to reduce the intermediate results
  if (num_blocks > 1) {
    size_t remaining = num_blocks;
    float *current_input = d_temp_result;

    while (remaining > 1) {
      int blocks_needed = get_num_blocks(remaining);
      sum_kernel<<<blocks_needed, BLOCK_SIZE, BLOCK_SIZE * sizeof(float), stream>>>(
          current_input, d_temp_result, remaining);
      remaining = blocks_needed;
    }
  }

  // Copy result back to host (SYNCHRONOUS)
  float result;
  cudaMemcpy(&result, d_temp_result, sizeof(float), cudaMemcpyDeviceToHost);

  // Clean up
  cudaFree(d_temp_result);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);

  return result;
}
// ... (cuda_dot_product, cuda_norm_squared, cuda_sum_squared_diff are kept synchronous) ...

// Host wrapper functions - Reduction operations for float
float cuda_dot_product(const float *a, const float *b, size_t size, cudaStream_t stream) {
  int num_blocks = get_num_blocks(size);

  float *d_temp_result;
  cudaMalloc(&d_temp_result, num_blocks * sizeof(float));

  dot_product_kernel<<<num_blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float), stream>>>(
      a, b, d_temp_result, size);
  if (num_blocks > 1) {
    size_t remaining = num_blocks;
    float *current_input = d_temp_result;

    while (remaining > 1) {
      int blocks_needed = get_num_blocks(remaining);
      sum_kernel<<<blocks_needed, BLOCK_SIZE, BLOCK_SIZE * sizeof(float), stream>>>(
          current_input, d_temp_result, remaining);
      remaining = blocks_needed;
    }
  }

  float result;
  cudaMemcpy(&result, d_temp_result, sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_temp_result);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);

  return result;
}

float cuda_norm_squared(const float *a, size_t size, cudaStream_t stream) {
  return cuda_dot_product(a, a, size, stream);
}

float cuda_sum_squared_diff(const float *a, float mean, size_t size, cudaStream_t stream) {
  int num_blocks = get_num_blocks(size);

  float *d_temp_result;
  cudaMalloc(&d_temp_result, num_blocks * sizeof(float));

  sum_squared_diff_kernel<<<num_blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float), stream>>>(
      a, mean, d_temp_result, size);

  if (num_blocks > 1) {
    size_t remaining = num_blocks;
    float *current_input = d_temp_result;

    while (remaining > 1) {
      int blocks_needed = get_num_blocks(remaining);
      sum_kernel<<<blocks_needed, BLOCK_SIZE, BLOCK_SIZE * sizeof(float), stream>>>(
          current_input, d_temp_result, remaining);
      remaining = blocks_needed;
    }
  }

  float result;
  cudaMemcpy(&result, d_temp_result, sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_temp_result);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);

  return result;
}

// Host wrapper functions - Reduction operations for double (These remain SYNCHRONOUS)
double cuda_sum(const double *a, size_t size, cudaStream_t stream) {
  int num_blocks = get_num_blocks(size);

  double *d_temp_result;
  cudaMalloc(&d_temp_result, num_blocks * sizeof(double));

  sum_kernel<<<num_blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(double), stream>>>(a, d_temp_result,
                                                                              size);

  if (num_blocks > 1) {
    size_t remaining = num_blocks;
    double *current_input = d_temp_result;

    while (remaining > 1) {
      int blocks_needed = get_num_blocks(remaining);
      sum_kernel<<<blocks_needed, BLOCK_SIZE, BLOCK_SIZE * sizeof(double), stream>>>(
          current_input, d_temp_result, remaining);
      remaining = blocks_needed;
    }
  }

  double result;
  cudaMemcpy(&result, d_temp_result, sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(d_temp_result);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);

  return result;
}

double cuda_dot_product(const double *a, const double *b, size_t size, cudaStream_t stream) {
  int num_blocks = get_num_blocks(size);

  double *d_temp_result;
  cudaMalloc(&d_temp_result, num_blocks * sizeof(double));

  dot_product_kernel<<<num_blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(double), stream>>>(
      a, b, d_temp_result, size);
  if (num_blocks > 1) {
    size_t remaining = num_blocks;
    double *current_input = d_temp_result;

    while (remaining > 1) {
      int blocks_needed = get_num_blocks(remaining);
      sum_kernel<<<blocks_needed, BLOCK_SIZE, BLOCK_SIZE * sizeof(double), stream>>>(
          current_input, d_temp_result, remaining);
      remaining = blocks_needed;
    }
  }

  double result;
  cudaMemcpy(&result, d_temp_result, sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(d_temp_result);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);

  return result;
}

double cuda_norm_squared(const double *a, size_t size, cudaStream_t stream) {
  return cuda_dot_product(a, a, size, stream);
}

double cuda_sum_squared_diff(const double *a, double mean, size_t size, cudaStream_t stream) {
  int num_blocks = get_num_blocks(size);

  double *d_temp_result;
  cudaMalloc(&d_temp_result, num_blocks * sizeof(double));

  sum_squared_diff_kernel<<<num_blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(double), stream>>>(
      a, mean, d_temp_result, size);

  if (num_blocks > 1) {
    size_t remaining = num_blocks;
    double *current_input = d_temp_result;

    while (remaining > 1) {
      int blocks_needed = get_num_blocks(remaining);
      sum_kernel<<<blocks_needed, BLOCK_SIZE, BLOCK_SIZE * sizeof(double), stream>>>(
          current_input, d_temp_result, remaining);
      remaining = blocks_needed;
    }
  }

  double result;
  cudaMemcpy(&result, d_temp_result, sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(d_temp_result);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);

  return result;
}

void cuda_sub_mul_scalar(const float *a, float sub_scalar, float mul_scalar, float *c, size_t size,
                         cudaStream_t stream) {
  int num_blocks = get_num_blocks(size);
  sub_mul_scalar_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(a, sub_scalar, mul_scalar, c, size);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

void cuda_mul_add_scalar(const float *a, float mul_scalar, float add_scalar, float *c, size_t size,
                         cudaStream_t stream) {
  int num_blocks = get_num_blocks(size);
  mul_add_scalar_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(a, mul_scalar, add_scalar, c, size);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

void cuda_sub_mul_scalar(const double *a, double sub_scalar, double mul_scalar, double *c,
                         size_t size, cudaStream_t stream) {
  int num_blocks = get_num_blocks(size);
  sub_mul_scalar_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(a, sub_scalar, mul_scalar, c, size);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

void cuda_mul_add_scalar(const double *a, double mul_scalar, double add_scalar, double *c,
                         size_t size, cudaStream_t stream) {
  int num_blocks = get_num_blocks(size);
  mul_add_scalar_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(a, mul_scalar, add_scalar, c, size);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

// Random number generation kernels (remain the same)
__global__ void fill_random_uniform_kernel(float *data, size_t size, float min_val, float max_val,
                                           unsigned long long seed) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    curandState state;
    curand_init(seed, idx, 0, &state);
    float random_val = curand_uniform(&state);
    data[idx] = min_val + random_val * (max_val - min_val);
  }
}

__global__ void fill_random_normal_kernel(float *data, size_t size, float mean, float stddev,
                                          unsigned long long seed) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    curandState state;
    curand_init(seed, idx, 0, &state);
    data[idx] = mean + stddev * curand_normal(&state);
  }
}

__global__ void fill_random_uniform_kernel(double *data, size_t size, double min_val,
                                           double max_val, unsigned long long seed) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    curandState state;
    curand_init(seed, idx, 0, &state);
    double random_val = curand_uniform_double(&state);
    data[idx] = min_val + random_val * (max_val - min_val);
  }
}

__global__ void fill_random_normal_kernel(double *data, size_t size, double mean, double stddev,
                                          unsigned long long seed) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    curandState state;
    curand_init(seed, idx, 0, &state);
    data[idx] = mean + stddev * curand_normal_double(&state);
  }
}

// Host wrapper functions - Random number generation for float
void cuda_fill_random_uniform(float *data, size_t size, float min_val, float max_val,
                              unsigned long long seed, cudaStream_t stream) {
  int num_blocks = get_num_blocks(size);
  fill_random_uniform_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(data, size, min_val, max_val,
                                                                    seed);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

void cuda_fill_random_normal(float *data, size_t size, float mean, float stddev,
                             unsigned long long seed, cudaStream_t stream) {
  int num_blocks = get_num_blocks(size);
  fill_random_normal_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(data, size, mean, stddev, seed);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

// Host wrapper functions - Random number generation for double
void cuda_fill_random_uniform(double *data, size_t size, double min_val, double max_val,
                              unsigned long long seed, cudaStream_t stream) {
  int num_blocks = get_num_blocks(size);
  fill_random_uniform_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(data, size, min_val, max_val,
                                                                    seed);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

void cuda_fill_random_normal(double *data, size_t size, double mean, double stddev,
                             unsigned long long seed, cudaStream_t stream) {
  int num_blocks = get_num_blocks(size);
  fill_random_normal_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(data, size, mean, stddev, seed);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

// Matrix transpose kernel (remains the same)
template <typename T>
__global__ void transpose_2d_kernel(const T *input, T *output, size_t rows, size_t cols) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < rows && col < cols) {
    int input_idx = row * cols + col;
    int output_idx = col * rows + row;
    output[output_idx] = input[input_idx];
  }
}

// NCHW to CNHW layout conversion kernel (remains the same)
template <typename T>
__global__ void nchw_to_cnhw_kernel(const T *input, T *output, size_t n, size_t c, size_t h,
                                    size_t w) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total_size = n * c * h * w;

  if (idx < total_size) {
    int n_idx = idx / (c * h * w);
    int remaining = idx % (c * h * w);
    int c_idx = remaining / (h * w);
    remaining = remaining % (h * w);
    int h_idx = remaining / w;
    int w_idx = remaining % w;

    int output_idx = c_idx * (n * h * w) + n_idx * (h * w) + h_idx * w + w_idx;
    output[output_idx] = input[idx];
  }
}

// CNHW to NCHW layout conversion kernel (remains the same)
template <typename T>
__global__ void cnhw_to_nchw_kernel(const T *input, T *output, size_t n, size_t c, size_t h,
                                    size_t w) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total_size = n * c * h * w;

  if (idx < total_size) {
    int c_idx = idx / (n * h * w);
    int remaining = idx % (n * h * w);
    int n_idx = remaining / (h * w);
    remaining = remaining % (h * w);
    int h_idx = remaining / w;
    int w_idx = remaining % w;

    int output_idx = n_idx * (c * h * w) + c_idx * (h * w) + h_idx * w + w_idx;
    output[output_idx] = input[idx];
  }
}

// Host wrapper functions - Template implementations
template <typename T>
void cuda_transpose_2d(const T *input, T *output, size_t rows, size_t cols, cudaStream_t stream) {
  dim3 block(16, 16);
  dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
  transpose_2d_kernel<<<grid, block, 0, stream>>>(input, output, rows, cols);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

template <typename T>
void cuda_nchw_to_cnhw(const T *input, T *output, size_t n, size_t c, size_t h, size_t w,
                       cudaStream_t stream) {
  size_t total_size = n * c * h * w;
  int num_blocks = get_num_blocks(total_size);
  nchw_to_cnhw_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(input, output, n, c, h, w);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

template <typename T>
void cuda_cnhw_to_nchw(const T *input, T *output, size_t n, size_t c, size_t h, size_t w,
                       cudaStream_t stream) {
  size_t total_size = n * c * h * w;
  int num_blocks = get_num_blocks(total_size);
  cnhw_to_nchw_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(input, output, n, c, h, w);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

// Explicit template instantiations (signatures updated)
template void cuda_transpose_2d<float>(const float *input, float *output, size_t rows, size_t cols,
                                       cudaStream_t stream);
template void cuda_transpose_2d<double>(const double *input, double *output, size_t rows,
                                        size_t cols, cudaStream_t stream);

template void cuda_nchw_to_cnhw<float>(const float *input, float *output, size_t n, size_t c,
                                       size_t h, size_t w, cudaStream_t stream);
template void cuda_nchw_to_cnhw<double>(const double *input, double *output, size_t n, size_t c,
                                        size_t h, size_t w, cudaStream_t stream);

template void cuda_cnhw_to_nchw<float>(const float *input, float *output, size_t n, size_t c,
                                       size_t h, size_t w, cudaStream_t stream);
template void cuda_cnhw_to_nchw<double>(const double *input, double *output, size_t n, size_t c,
                                        size_t h, size_t w, cudaStream_t stream);

} // namespace cuda

#endif // USE_CUDA
} // namespace tnn