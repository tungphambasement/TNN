#pragma once

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#endif

#include <cmath>
#include <cstddef>

namespace tnn {
namespace cuda {

// Basic arithmetic operations
void cuda_add(const float *a, const float *b, float *c, size_t size, cudaStream_t stream);
void cuda_sub(const float *a, const float *b, float *c, size_t size, cudaStream_t stream);
void cuda_mul(const float *a, const float *b, float *c, size_t size, cudaStream_t stream);
void cuda_div(const float *a, const float *b, float *c, size_t size, cudaStream_t stream);

void cuda_add(const double *a, const double *b, double *c, size_t size, cudaStream_t stream);
void cuda_sub(const double *a, const double *b, double *c, size_t size, cudaStream_t stream);
void cuda_mul(const double *a, const double *b, double *c, size_t size, cudaStream_t stream);
void cuda_div(const double *a, const double *b, double *c, size_t size, cudaStream_t stream);

// Fused multiply-add operations
void cuda_fmadd(const float *a, const float *b, float *c, size_t size, cudaStream_t stream);
void cuda_fmsub(const float *a, const float *b, float *c, size_t size, cudaStream_t stream);
void cuda_fnmadd(const float *a, const float *b, float *c, size_t size, cudaStream_t stream);

void cuda_fmadd(const double *a, const double *b, double *c, size_t size, cudaStream_t stream);
void cuda_fmsub(const double *a, const double *b, double *c, size_t size, cudaStream_t stream);
void cuda_fnmadd(const double *a, const double *b, double *c, size_t size, cudaStream_t stream);

// Scalar operations
void cuda_add_scalar(const float *a, float scalar, float *c, size_t size, cudaStream_t stream);
void cuda_mul_scalar(const float *a, float scalar, float *c, size_t size, cudaStream_t stream);
void cuda_div_scalar(const float *a, float scalar, float *c, size_t size, cudaStream_t stream);
void cuda_set_scalar(float *c, float scalar, size_t size, cudaStream_t stream);

void cuda_add_scalar(const double *a, double scalar, double *c, size_t size, cudaStream_t stream);
void cuda_mul_scalar(const double *a, double scalar, double *c, size_t size, cudaStream_t stream);
void cuda_div_scalar(const double *a, double scalar, double *c, size_t size, cudaStream_t stream);
void cuda_set_scalar(double *c, double scalar, size_t size, cudaStream_t stream);

// Mathematical functions
void cuda_sqrt(const float *a, float *c, size_t size, cudaStream_t stream);
void cuda_rsqrt(const float *a, float *c, size_t size, cudaStream_t stream);
void cuda_rcp(const float *a, float *c, size_t size, cudaStream_t stream);
void cuda_abs(const float *a, float *c, size_t size, cudaStream_t stream);

void cuda_sqrt(const double *a, double *c, size_t size, cudaStream_t stream);
void cuda_abs(const double *a, double *c, size_t size, cudaStream_t stream);

// Min/Max operations
void cuda_min(const float *a, const float *b, float *c, size_t size, cudaStream_t stream);
void cuda_max(const float *a, const float *b, float *c, size_t size, cudaStream_t stream);
void cuda_scalar_max(const float *a, float scalar, float *c, size_t size, cudaStream_t stream);
void cuda_clamp(const float *a, float min_val, float max_val, float *c, size_t size,
                cudaStream_t stream);

void cuda_min(const double *a, const double *b, double *c, size_t size, cudaStream_t stream);
void cuda_max(const double *a, const double *b, double *c, size_t size, cudaStream_t stream);
void cuda_scalar_max(const double *a, double scalar, double *c, size_t size, cudaStream_t stream);
void cuda_clamp(const double *a, double min_val, double max_val, double *c, size_t size,
                cudaStream_t stream);

// Comparison operations
void cuda_equal(const float *a, const float *b, float *c, size_t size, cudaStream_t stream);
void cuda_greater(const float *a, const float *b, float *c, size_t size, cudaStream_t stream);

void cuda_equal(const double *a, const double *b, double *c, size_t size, cudaStream_t stream);
void cuda_greater(const double *a, const double *b, double *c, size_t size, cudaStream_t stream);

// Memory operations
void cuda_copy(const float *a, float *c, size_t size, cudaStream_t stream);
void cuda_zero(float *c, size_t size, cudaStream_t stream);

void cuda_copy(const double *a, double *c, size_t size, cudaStream_t stream);
void cuda_zero(double *c, size_t size, cudaStream_t stream);

// Reduction operations
float cuda_sum(const float *a, size_t size, cudaStream_t stream);
float cuda_dot_product(const float *a, const float *b, size_t size, cudaStream_t stream);
float cuda_norm_squared(const float *a, size_t size, cudaStream_t stream);
float cuda_sum_squared_diff(const float *a, float mean, size_t size, cudaStream_t stream);

double cuda_sum(const double *a, size_t size, cudaStream_t stream);
double cuda_dot_product(const double *a, const double *b, size_t size, cudaStream_t stream);
double cuda_norm_squared(const double *a, size_t size, cudaStream_t stream);
double cuda_sum_squared_diff(const double *a, double mean, size_t size, cudaStream_t stream);

// Specialized BatchNorm operations
void cuda_sub_mul_scalar(const float *a, float sub_scalar, float mul_scalar, float *c, size_t size,
                         cudaStream_t stream);
void cuda_mul_add_scalar(const float *a, float mul_scalar, float add_scalar, float *c, size_t size,
                         cudaStream_t stream);

void cuda_sub_mul_scalar(const double *a, double sub_scalar, double mul_scalar, double *c,
                         size_t size, cudaStream_t stream);
void cuda_mul_add_scalar(const double *a, double mul_scalar, double add_scalar, double *c,
                         size_t size, cudaStream_t stream);

// Random number generation operations
void cuda_fill_random_uniform(float *data, size_t size, float min_val, float max_val,
                              unsigned long long seed, cudaStream_t stream);
void cuda_fill_random_normal(float *data, size_t size, float mean, float stddev,
                             unsigned long long seed, cudaStream_t stream);

void cuda_fill_random_uniform(double *data, size_t size, double min_val, double max_val,
                              unsigned long long seed, cudaStream_t stream);
void cuda_fill_random_normal(double *data, size_t size, double mean, double stddev,
                             unsigned long long seed, cudaStream_t stream);

template <typename T>
void cuda_transpose_2d(const T *input, T *output, size_t rows, size_t cols, cudaStream_t stream);

template <typename T>
void cuda_nchw_to_cnhw(const T *input, T *output, size_t n, size_t c, size_t h, size_t w,
                       cudaStream_t stream);

template <typename T>
void cuda_cnhw_to_nchw(const T *input, T *output, size_t n, size_t c, size_t h, size_t w,
                       cudaStream_t stream);

} // namespace cuda
} // namespace tnn