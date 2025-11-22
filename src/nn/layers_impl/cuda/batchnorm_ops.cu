/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/cuda/batchnorm_ops.hpp"

#include <cuda_runtime.h>

namespace tnn {
namespace cuda {
namespace batchnorm {

#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32

// Warp-level reduction primitives for better performance
template <typename T> __inline__ __device__ T warpReduceSum(T val) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

template <typename T> __inline__ __device__ T blockReduceSum(T val) {
  static __shared__ T shared[WARP_SIZE];
  int lane = threadIdx.x % WARP_SIZE;
  int wid = threadIdx.x / WARP_SIZE;

  val = warpReduceSum(val);

  if (lane == 0)
    shared[wid] = val;
  __syncthreads();

  val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : T(0);
  if (wid == 0)
    val = warpReduceSum(val);

  return val;
}

template <typename T>
__global__ void compute_channel_mean_kernel(const T *input_data, T *mean_data, size_t batch_size,
                                            size_t channels, size_t spatial_size) {

  const int c = blockIdx.x;
  if (c >= channels)
    return;

  const size_t total_elements_per_channel = batch_size * spatial_size;
  const T inv_total = T(1) / static_cast<T>(total_elements_per_channel);
  const size_t channel_stride = channels * spatial_size;
  const size_t c_offset = c * spatial_size;

  T thread_sum = T(0);

  for (size_t i = threadIdx.x; i < total_elements_per_channel; i += blockDim.x) {
    const size_t n = i / spatial_size;
    const size_t i_local = i % spatial_size;
    const size_t global_idx = n * channel_stride + c_offset + i_local;
    thread_sum += input_data[global_idx];
  }

  thread_sum = blockReduceSum(thread_sum);

  if (threadIdx.x == 0) {
    mean_data[c] = thread_sum * inv_total;
  }
}

template <typename T>
__global__ void compute_channel_variance_kernel(const T *input_data, const T *mean_data,
                                                T *var_data, size_t batch_size, size_t channels,
                                                size_t spatial_size) {

  const int c = blockIdx.x;
  if (c >= channels)
    return;

  const size_t total_elements_per_channel = batch_size * spatial_size;
  const T inv_total = T(1) / static_cast<T>(total_elements_per_channel);
  const T mean_val = mean_data[c];
  const size_t channel_stride = channels * spatial_size;
  const size_t c_offset = c * spatial_size;

  T thread_sum_sq = T(0);

  for (size_t i = threadIdx.x; i < total_elements_per_channel; i += blockDim.x) {
    const size_t n = i / spatial_size;
    const size_t i_local = i % spatial_size;
    const size_t global_idx = n * channel_stride + c_offset + i_local;

    T diff = input_data[global_idx] - mean_val;
    thread_sum_sq += diff * diff;
  }

  thread_sum_sq = blockReduceSum(thread_sum_sq);

  if (threadIdx.x == 0) {
    var_data[c] = thread_sum_sq * inv_total;
  }
}

template <typename T>
__global__ void compute_affine_gradients_kernel(const T *gradient_data, const T *normalized_data,
                                                T *gamma_grad, T *beta_grad, size_t batch_size,
                                                size_t channels, size_t spatial_size) {

  int c = blockIdx.x;
  if (c >= channels)
    return;

  const size_t total_elements_per_channel = batch_size * spatial_size;
  const size_t channel_stride = channels * spatial_size;
  const size_t c_offset = c * spatial_size;

  T thread_gamma_sum = T(0);
  T thread_beta_sum = T(0);

  for (size_t i = threadIdx.x; i < total_elements_per_channel; i += blockDim.x) {
    const size_t n = i / spatial_size;
    const size_t i_local = i % spatial_size;
    const size_t idx = n * channel_stride + c_offset + i_local;

    thread_gamma_sum += gradient_data[idx] * normalized_data[idx];
    thread_beta_sum += gradient_data[idx];
  }

  thread_gamma_sum = blockReduceSum(thread_gamma_sum);
  thread_beta_sum = blockReduceSum(thread_beta_sum);

  if (threadIdx.x == 0) {
    atomicAdd(&gamma_grad[c], thread_gamma_sum);
    atomicAdd(&beta_grad[c], thread_beta_sum);
  }
}

template <typename T>
__global__ void normalize_and_scalekernel(const T *input_data, const T *mean_data,
                                          const T *std_data, const T *gamma_data,
                                          const T *beta_data, T *output_data, T *normalized_data,
                                          size_t batch_size, size_t channels, size_t spatial_size,
                                          bool affine) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_elements = batch_size * channels * spatial_size;

  if (idx >= total_elements)
    return;

  int c = (idx / spatial_size) % channels;

  const T mean_val = mean_data[c];
  const T std_val = std_data[c];
  const T inv_std = T(1) / std_val;

  T input_val = input_data[idx];
  T normalized_val = (input_val - mean_val) * inv_std;
  normalized_data[idx] = normalized_val;

  if (affine) {
    const T gamma_val = gamma_data[c];
    const T beta_val = beta_data[c];
    output_data[idx] = gamma_val * normalized_val + beta_val;
  } else {
    output_data[idx] = normalized_val;
  }
}

template <typename T>
__global__ void compute_batch_std_kernel(const T *batch_var_data, T *batch_std_data,
                                         size_t channels, T epsilon) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < channels) {
    batch_std_data[c] = sqrt(batch_var_data[c] + epsilon);
  }
}

// Fused mean and variance computation -  with warp primitives
template <typename T>
__global__ void compute_mean_variance_fused_kernel(const T *input_data, T *mean_data, T *var_data,
                                                   size_t batch_size, size_t channels,
                                                   size_t spatial_size) {
  const int c = blockIdx.x;
  if (c >= channels)
    return;

  const size_t total_elements_per_channel = batch_size * spatial_size;
  const size_t channel_stride = channels * spatial_size;
  const size_t c_offset = c * spatial_size;

  // Single pass: compute sum and sum of squares simultaneously
  T thread_sum = T(0);
  T thread_sum_sq = T(0);

  //  memory access - stride by block size
  for (size_t i = threadIdx.x; i < total_elements_per_channel; i += blockDim.x) {
    const size_t n = i / spatial_size;
    const size_t i_local = i % spatial_size;
    const size_t global_idx = n * channel_stride + c_offset + i_local;
    const T val = input_data[global_idx];
    thread_sum += val;
    thread_sum_sq += val * val;
  }

  // Use warp-level reduction instead of shared memory
  thread_sum = blockReduceSum(thread_sum);
  thread_sum_sq = blockReduceSum(thread_sum_sq);

  if (threadIdx.x == 0) {
    const T inv_n = T(1) / static_cast<T>(total_elements_per_channel);
    const T mean_val = thread_sum * inv_n;
    const T mean_sq = thread_sum_sq * inv_n;
    mean_data[c] = mean_val;
    var_data[c] = mean_sq - mean_val * mean_val;
  }
}

template <typename T>
void compute_mean_variance_fused(const T *input_data, T *mean_data, T *var_data, size_t batch_size,
                                 size_t channels, size_t spatial_size, cudaStream_t stream) {
  int threads_per_block = THREADS_PER_BLOCK;
  int num_blocks = channels;

  compute_mean_variance_fused_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
      input_data, mean_data, var_data, batch_size, channels, spatial_size);
}

template <typename T>
__global__ void update_running_stats_kernel(T *running_mean_data, T *running_var_data,
                                            const T *batch_mean_data, const T *batch_var_data,
                                            size_t channels, T momentum) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < channels) {
    running_mean_data[c] = (T(1) - momentum) * running_mean_data[c] + momentum * batch_mean_data[c];
    running_var_data[c] = (T(1) - momentum) * running_var_data[c] + momentum * batch_var_data[c];
  }
}

template <typename T>
__global__ void compute_inference_output_kernel(const T *input_data, const T *running_mean_data,
                                                const T *running_var_data, const T *gamma_data,
                                                const T *beta_data, T *output_data,
                                                size_t batch_size, size_t channels,
                                                size_t spatial_size, T epsilon, bool affine) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_elements = batch_size * channels * spatial_size;

  if (idx >= total_elements)
    return;

  int c = (idx / spatial_size) % channels;

  T mean_val = running_mean_data[c];
  T var_val = running_var_data[c];
  T std_val = sqrt(var_val + epsilon);
  const T inv_std = T(1) / std_val;

  T input_val = input_data[idx];
  T normalized_val = (input_val - mean_val) * inv_std;

  if (affine) {
    const T gamma_val = gamma_data[c];
    const T beta_val = beta_data[c];
    output_data[idx] = gamma_val * normalized_val + beta_val;
  } else {
    output_data[idx] = normalized_val;
  }
}

template <typename T>
__global__ void compute_grad_normalized_kernel(const T *gradient_data, const T *gamma_data,
                                               T *grad_normalized_data, size_t batch_size,
                                               size_t channels, size_t spatial_size, bool affine) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_elements = batch_size * channels * spatial_size;

  if (idx >= total_elements)
    return;

  if (affine) {
    int c = (idx / spatial_size) % channels;
    const T gamma_val = gamma_data[c];
    grad_normalized_data[idx] = gradient_data[idx] * gamma_val;
  } else {
    grad_normalized_data[idx] = gradient_data[idx];
  }
}

template <typename T>
__global__ void compute_backward_sums_kernel(const T *grad_normalized_data,
                                             const T *normalized_data, T *sum_grad_normalized_data,
                                             T *sum_grad_norm_times_norm_data, size_t batch_size,
                                             size_t channels, size_t spatial_size) {

  const int c = blockIdx.x;
  if (c >= channels)
    return;

  const size_t total_elements_per_channel = batch_size * spatial_size;
  const size_t channel_stride = channels * spatial_size;
  const size_t c_offset = c * spatial_size;

  T thread_sum_grad_norm = T(0);
  T thread_sum_grad_norm_x_norm = T(0);

  for (size_t i = threadIdx.x; i < total_elements_per_channel; i += blockDim.x) {
    const size_t n = i / spatial_size;
    const size_t i_local = i % spatial_size;
    const size_t global_idx = n * channel_stride + c_offset + i_local;

    thread_sum_grad_norm += grad_normalized_data[global_idx];
    thread_sum_grad_norm_x_norm += grad_normalized_data[global_idx] * normalized_data[global_idx];
  }

  thread_sum_grad_norm = blockReduceSum(thread_sum_grad_norm);
  thread_sum_grad_norm_x_norm = blockReduceSum(thread_sum_grad_norm_x_norm);

  if (threadIdx.x == 0) {
    sum_grad_normalized_data[c] = thread_sum_grad_norm;
    sum_grad_norm_times_norm_data[c] = thread_sum_grad_norm_x_norm;
  }
}

template <typename T>
__global__ void compute_input_gradients_batchnorm_kernel(
    const T *grad_normalized_data, const T *normalized_data, const T *std_data,
    const T *sum_grad_normalized_data, const T *sum_grad_norm_times_norm_data, T *grad_input_data,
    size_t batch_size, size_t channels, size_t spatial_size, size_t total_elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_size = batch_size * channels * spatial_size;

  if (idx >= total_size)
    return;

  int c = (idx / spatial_size) % channels;

  const T std_val_c = std_data[c];
  const T inv_std = T(1) / std_val_c;
  const T sum_grad_norm = sum_grad_normalized_data[c];
  const T sum_grad_norm_x_norm = sum_grad_norm_times_norm_data[c];
  const T inv_total = T(1) / static_cast<T>(total_elements);

  grad_input_data[idx] = inv_std * inv_total *
                         (static_cast<T>(total_elements) * grad_normalized_data[idx] -
                          sum_grad_norm - normalized_data[idx] * sum_grad_norm_x_norm);
}

// Fused backward kernel: combines affine gradients + backward sums computation -
template <typename T>
__global__ void compute_batchnorm_backward_sums_fused_kernel(
    const T *gradient_data, const T *normalized_data, const T *gamma_data,
    T *sum_grad_normalized_data, T *sum_grad_norm_times_norm_data, T *gamma_grad, T *beta_grad,
    size_t batch_size, size_t channels, size_t spatial_size, bool affine) {

  const int c = blockIdx.x;
  if (c >= channels)
    return;

  const size_t total_elements_per_channel = batch_size * spatial_size;
  const size_t channel_stride = channels * spatial_size;
  const size_t c_offset = c * spatial_size;

  T thread_sum_grad_norm = T(0);
  T thread_sum_grad_norm_x_norm = T(0);
  T thread_gamma_sum = T(0);
  T thread_beta_sum = T(0);

  const T gamma_val = affine ? gamma_data[c] : T(1);

  //  loop with better memory access
  for (size_t i = threadIdx.x; i < total_elements_per_channel; i += blockDim.x) {
    const size_t n = i / spatial_size;
    const size_t i_local = i % spatial_size;
    const size_t global_idx = n * channel_stride + c_offset + i_local;

    const T grad_val = gradient_data[global_idx];
    const T norm_val = normalized_data[global_idx];
    const T grad_norm_val = grad_val * gamma_val;

    thread_sum_grad_norm += grad_norm_val;
    thread_sum_grad_norm_x_norm += grad_norm_val * norm_val;

    if (affine) {
      thread_gamma_sum += grad_val * norm_val;
      thread_beta_sum += grad_val;
    }
  }

  // Use warp-level reductions instead of shared memory
  thread_sum_grad_norm = blockReduceSum(thread_sum_grad_norm);
  thread_sum_grad_norm_x_norm = blockReduceSum(thread_sum_grad_norm_x_norm);

  if (affine) {
    thread_gamma_sum = blockReduceSum(thread_gamma_sum);
    thread_beta_sum = blockReduceSum(thread_beta_sum);
  }

  if (threadIdx.x == 0) {
    sum_grad_normalized_data[c] = thread_sum_grad_norm;
    sum_grad_norm_times_norm_data[c] = thread_sum_grad_norm_x_norm;
    if (affine) {
      atomicAdd(&gamma_grad[c], thread_gamma_sum);
      atomicAdd(&beta_grad[c], thread_beta_sum);
    }
  }
}

// Fused input gradients kernel
template <typename T>
__global__ void compute_input_gradients_batchnorm_fused_kernel(
    const T *gradient_data, const T *normalized_data, const T *std_data, const T *gamma_data,
    const T *sum_grad_normalized_data, const T *sum_grad_norm_times_norm_data, T *grad_input_data,
    size_t batch_size, size_t channels, size_t spatial_size, size_t total_elements, bool affine) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_size = batch_size * channels * spatial_size;

  if (idx >= total_size)
    return;

  int c = (idx / spatial_size) % channels;

  const T std_val_c = std_data[c];
  const T inv_std = T(1) / std_val_c;
  const T sum_grad_norm = sum_grad_normalized_data[c];
  const T sum_grad_norm_x_norm = sum_grad_norm_times_norm_data[c];
  const T inv_total = T(1) / static_cast<T>(total_elements);
  const T gamma_val = affine ? gamma_data[c] : T(1);

  const T grad_norm_val = gradient_data[idx] * gamma_val;

  grad_input_data[idx] = inv_std * inv_total *
                         (static_cast<T>(total_elements) * grad_norm_val - sum_grad_norm -
                          normalized_data[idx] * sum_grad_norm_x_norm);
}

template <typename T>
void compute_batchnorm_backward_fused(const T *gradient_data, const T *normalized_data,
                                      const T *std_data, const T *gamma_data, T *grad_input_data,
                                      T *gamma_grad, T *beta_grad, size_t batch_size,
                                      size_t channels, size_t spatial_size, bool affine,
                                      cudaStream_t stream) {
  const size_t total_elements = batch_size * spatial_size;

  // Allocate temporary buffers - use cudaMallocAsync for stream ordering
  T *sum_grad_normalized_data;
  T *sum_grad_norm_times_norm_data;
  cudaMallocAsync(&sum_grad_normalized_data, channels * sizeof(T), stream);
  cudaMallocAsync(&sum_grad_norm_times_norm_data, channels * sizeof(T), stream);

  // First pass: compute sums and affine gradients with  kernel (no large shared mem)
  int threads_per_block = THREADS_PER_BLOCK;
  int num_blocks = channels;

  compute_batchnorm_backward_sums_fused_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
      gradient_data, normalized_data, gamma_data, sum_grad_normalized_data,
      sum_grad_norm_times_norm_data, gamma_grad, beta_grad, batch_size, channels, spatial_size,
      affine);

  // Second pass: compute input gradients
  size_t total_size = batch_size * channels * spatial_size;
  num_blocks = (total_size + threads_per_block - 1) / threads_per_block;

  compute_input_gradients_batchnorm_fused_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
      gradient_data, normalized_data, std_data, gamma_data, sum_grad_normalized_data,
      sum_grad_norm_times_norm_data, grad_input_data, batch_size, channels, spatial_size,
      total_elements, affine);

  cudaFreeAsync(sum_grad_normalized_data, stream);
  cudaFreeAsync(sum_grad_norm_times_norm_data, stream);
}

template <typename T>
void compute_channel_mean(const T *input_data, T *mean_data, size_t batch_size, size_t channels,
                          size_t spatial_size, cudaStream_t stream) {
  int threads_per_block = THREADS_PER_BLOCK;
  int num_blocks = channels;

  compute_channel_mean_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
      input_data, mean_data, batch_size, channels, spatial_size);
}

template <typename T>
void compute_channel_variance(const T *input_data, const T *mean_data, T *var_data,
                              size_t batch_size, size_t channels, size_t spatial_size,
                              cudaStream_t stream) {
  int threads_per_block = THREADS_PER_BLOCK;
  int num_blocks = channels;

  compute_channel_variance_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
      input_data, mean_data, var_data, batch_size, channels, spatial_size);
}

template <typename T>
void normalize_and_scale(const T *input_data, const T *mean_data, const T *std_data,
                         const T *gamma_data, const T *beta_data, T *output_data,
                         T *normalized_data, size_t batch_size, size_t channels,
                         size_t spatial_size, bool affine, cudaStream_t stream) {
  size_t total_elements = batch_size * channels * spatial_size;
  int threads_per_block = THREADS_PER_BLOCK;
  int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

  normalize_and_scalekernel<<<num_blocks, threads_per_block, 0, stream>>>(
      input_data, mean_data, std_data, gamma_data, beta_data, output_data, normalized_data,
      batch_size, channels, spatial_size, affine);
}

template <typename T>
void compute_affine_gradients_(const T *gradient_data, const T *normalized_data, T *gamma_grad,
                               T *beta_grad, size_t batch_size, size_t channels,
                               size_t spatial_size, cudaStream_t stream) {
  int threads_per_block = THREADS_PER_BLOCK;
  int num_blocks = channels;

  compute_affine_gradients_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
      gradient_data, normalized_data, gamma_grad, beta_grad, batch_size, channels, spatial_size);
}

template <typename T>
void compute_batch_std(const T *batch_var_data, T *batch_std_data, size_t channels, T epsilon,
                       cudaStream_t stream) {
  int threads_per_block = THREADS_PER_BLOCK;
  int num_blocks = (channels + threads_per_block - 1) / threads_per_block;

  compute_batch_std_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
      batch_var_data, batch_std_data, channels, epsilon);
}

template <typename T>
void update_running_stats(T *running_mean_data, T *running_var_data, const T *batch_mean_data,
                          const T *batch_var_data, size_t channels, T momentum,
                          cudaStream_t stream) {
  int threads_per_block = THREADS_PER_BLOCK;
  int num_blocks = (channels + threads_per_block - 1) / threads_per_block;

  update_running_stats_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
      running_mean_data, running_var_data, batch_mean_data, batch_var_data, channels, momentum);
}

template <typename T>
void compute_inference_output(const T *input_data, const T *running_mean_data,
                              const T *running_var_data, const T *gamma_data, const T *beta_data,
                              T *output_data, size_t batch_size, size_t channels,
                              size_t spatial_size, T epsilon, bool affine, cudaStream_t stream) {
  size_t total_elements = batch_size * channels * spatial_size;
  int threads_per_block = THREADS_PER_BLOCK;
  int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

  compute_inference_output_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
      input_data, running_mean_data, running_var_data, gamma_data, beta_data, output_data,
      batch_size, channels, spatial_size, epsilon, affine);
}

template <typename T>
void compute_grad_normalized(const T *gradient_data, const T *gamma_data, T *grad_normalized_data,
                             size_t batch_size, size_t channels, size_t spatial_size, bool affine,
                             cudaStream_t stream) {
  size_t total_elements = batch_size * channels * spatial_size;
  int threads_per_block = THREADS_PER_BLOCK;
  int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

  compute_grad_normalized_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
      gradient_data, gamma_data, grad_normalized_data, batch_size, channels, spatial_size, affine);
}

template <typename T>
void compute_backward_sums(const T *grad_normalized_data, const T *normalized_data,
                           T *sum_grad_normalized_data, T *sum_grad_norm_times_norm_data,
                           size_t batch_size, size_t channels, size_t spatial_size,
                           cudaStream_t stream) {

  int threads_per_block = THREADS_PER_BLOCK;
  int num_blocks = channels;

  compute_backward_sums_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
      grad_normalized_data, normalized_data, sum_grad_normalized_data,
      sum_grad_norm_times_norm_data, batch_size, channels, spatial_size);
}

template <typename T>
void compute_input_gradients_batchnorm(const T *grad_normalized_data, const T *normalized_data,
                                       const T *std_data, const T *sum_grad_normalized_data,
                                       const T *sum_grad_norm_times_norm_data, T *grad_input_data,
                                       size_t batch_size, size_t channels, size_t spatial_size,
                                       size_t total_elements, cudaStream_t stream) {
  size_t total_size = batch_size * channels * spatial_size;
  int threads_per_block = THREADS_PER_BLOCK;
  int num_blocks = (total_size + threads_per_block - 1) / threads_per_block;

  compute_input_gradients_batchnorm_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
      grad_normalized_data, normalized_data, std_data, sum_grad_normalized_data,
      sum_grad_norm_times_norm_data, grad_input_data, batch_size, channels, spatial_size,
      total_elements);
}

template void compute_channel_mean<float>(const float *input_data, float *mean_data,
                                          size_t batch_size, size_t channels, size_t spatial_size,
                                          cudaStream_t stream);
template void compute_channel_mean<double>(const double *input_data, double *mean_data,
                                           size_t batch_size, size_t channels, size_t spatial_size,
                                           cudaStream_t stream);

template void compute_channel_variance<float>(const float *input_data, const float *mean_data,
                                              float *var_data, size_t batch_size, size_t channels,
                                              size_t spatial_size, cudaStream_t stream);
template void compute_channel_variance<double>(const double *input_data, const double *mean_data,
                                               double *var_data, size_t batch_size, size_t channels,
                                               size_t spatial_size, cudaStream_t stream);

template void compute_mean_variance_fused<float>(const float *input_data, float *mean_data,
                                                 float *var_data, size_t batch_size,
                                                 size_t channels, size_t spatial_size,
                                                 cudaStream_t stream);
template void compute_mean_variance_fused<double>(const double *input_data, double *mean_data,
                                                  double *var_data, size_t batch_size,
                                                  size_t channels, size_t spatial_size,
                                                  cudaStream_t stream);

template void normalize_and_scale<float>(const float *input_data, const float *mean_data,
                                         const float *std_data, const float *gamma_data,
                                         const float *beta_data, float *output_data,
                                         float *normalized_data, size_t batch_size, size_t channels,
                                         size_t spatial_size, bool affine, cudaStream_t stream);
template void normalize_and_scale<double>(const double *input_data, const double *mean_data,
                                          const double *std_data, const double *gamma_data,
                                          const double *beta_data, double *output_data,
                                          double *normalized_data, size_t batch_size,
                                          size_t channels, size_t spatial_size, bool affine,
                                          cudaStream_t stream);

template void compute_batch_std<float>(const float *batch_var_data, float *batch_std_data,
                                       size_t channels, float epsilon, cudaStream_t stream);
template void compute_batch_std<double>(const double *batch_var_data, double *batch_std_data,
                                        size_t channels, double epsilon, cudaStream_t stream);

template void update_running_stats<float>(float *running_mean_data, float *running_var_data,
                                          const float *batch_mean_data, const float *batch_var_data,
                                          size_t channels, float momentum, cudaStream_t stream);
template void update_running_stats<double>(double *running_mean_data, double *running_var_data,
                                           const double *batch_mean_data,
                                           const double *batch_var_data, size_t channels,
                                           double momentum, cudaStream_t stream);

template void compute_inference_output<float>(
    const float *input_data, const float *running_mean_data, const float *running_var_data,
    const float *gamma_data, const float *beta_data, float *output_data, size_t batch_size,
    size_t channels, size_t spatial_size, float epsilon, bool affine, cudaStream_t stream);
template void compute_inference_output<double>(
    const double *input_data, const double *running_mean_data, const double *running_var_data,
    const double *gamma_data, const double *beta_data, double *output_data, size_t batch_size,
    size_t channels, size_t spatial_size, double epsilon, bool affine, cudaStream_t stream);

template void compute_affine_gradients_<float>(const float *gradient_data,
                                               const float *normalized_data, float *gamma_grad,
                                               float *beta_grad, size_t batch_size, size_t channels,
                                               size_t spatial_size, cudaStream_t stream);
template void compute_affine_gradients_<double>(const double *gradient_data,
                                                const double *normalized_data, double *gamma_grad,
                                                double *beta_grad, size_t batch_size,
                                                size_t channels, size_t spatial_size,
                                                cudaStream_t stream);

template void compute_grad_normalized<float>(const float *gradient_data, const float *gamma_data,
                                             float *grad_normalized_data, size_t batch_size,
                                             size_t channels, size_t spatial_size, bool affine,
                                             cudaStream_t stream);
template void compute_grad_normalized<double>(const double *gradient_data, const double *gamma_data,
                                              double *grad_normalized_data, size_t batch_size,
                                              size_t channels, size_t spatial_size, bool affine,
                                              cudaStream_t stream);

template void compute_backward_sums<float>(const float *grad_normalized_data,
                                           const float *normalized_data,
                                           float *sum_grad_normalized_data,
                                           float *sum_grad_norm_times_norm_data, size_t batch_size,
                                           size_t channels, size_t spatial_size,
                                           cudaStream_t stream);
template void compute_backward_sums<double>(const double *grad_normalized_data,
                                            const double *normalized_data,
                                            double *sum_grad_normalized_data,
                                            double *sum_grad_norm_times_norm_data,
                                            size_t batch_size, size_t channels, size_t spatial_size,
                                            cudaStream_t stream);

template void compute_input_gradients_batchnorm<float>(
    const float *grad_normalized_data, const float *normalized_data, const float *std_data,
    const float *sum_grad_normalized_data, const float *sum_grad_norm_times_norm_data,
    float *grad_input_data, size_t batch_size, size_t channels, size_t spatial_size,
    size_t total_elements, cudaStream_t stream);
template void compute_input_gradients_batchnorm<double>(
    const double *grad_normalized_data, const double *normalized_data, const double *std_data,
    const double *sum_grad_normalized_data, const double *sum_grad_norm_times_norm_data,
    double *grad_input_data, size_t batch_size, size_t channels, size_t spatial_size,
    size_t total_elements, cudaStream_t stream);

template void compute_batchnorm_backward_fused<float>(
    const float *gradient_data, const float *normalized_data, const float *std_data,
    const float *gamma_data, float *grad_input_data, float *gamma_grad, float *beta_grad,
    size_t batch_size, size_t channels, size_t spatial_size, bool affine, cudaStream_t stream);
template void compute_batchnorm_backward_fused<double>(
    const double *gradient_data, const double *normalized_data, const double *std_data,
    const double *gamma_data, double *grad_input_data, double *gamma_grad, double *beta_grad,
    size_t batch_size, size_t channels, size_t spatial_size, bool affine, cudaStream_t stream);

} // namespace batchnorm
} // namespace cuda
} // namespace tnn