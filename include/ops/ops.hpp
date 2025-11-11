#pragma once

#include "cpu/kernels.hpp"
#include "device/device_ptr.hpp"
#ifdef USE_CUDA
#include "cuda/kernels.hpp"
#endif

#include <cstddef>
#include <stdexcept>

namespace tnn {
namespace ops {

// Arithmetic Operations
template <typename T>
void add(const device_ptr<T[]> &a, const device_ptr<T[]> &b, device_ptr<T[]> &c, size_t size) {
  if (!a.getDevice() || !b.getDevice() || !c.getDevice()) {
    throw std::runtime_error("Device pointer has no associated device");
  }
  if (a.getDevice() != b.getDevice() || a.getDevice() != c.getDevice()) {
    throw std::runtime_error("All device pointers must be on the same device");
  }
  auto device_type = a.getDevice()->getDeviceType();

  if (device_type == DeviceType::CPU) {
    cpu::add(a.get(), b.get(), c.get(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    cuda::cuda_add(a.get(), b.get(), c.get(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
void sub(const device_ptr<T[]> &a, const device_ptr<T[]> &b, device_ptr<T[]> &c, size_t size) {
  if (!a.getDevice() || !b.getDevice() || !c.getDevice()) {
    throw std::runtime_error("Device pointer has no associated device");
  }
  if (a.getDevice() != b.getDevice() || a.getDevice() != c.getDevice()) {
    throw std::runtime_error("All device pointers must be on the same device");
  }

  auto device_type = a.getDevice()->getDeviceType();

  if (device_type == DeviceType::CPU) {
    cpu::sub(a.get(), b.get(), c.get(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    cuda::cuda_sub(a.get(), b.get(), c.get(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
void mul(const device_ptr<T[]> &a, const device_ptr<T[]> &b, device_ptr<T[]> &c, size_t size) {
  if (!a.getDevice() || !b.getDevice() || !c.getDevice()) {
    throw std::runtime_error("Device pointer has no associated device");
  }
  if (a.getDevice() != b.getDevice() || a.getDevice() != c.getDevice()) {
    throw std::runtime_error("All device pointers must be on the same device");
  }

  auto device_type = a.getDevice()->getDeviceType();

  if (device_type == DeviceType::CPU) {
    cpu::mul(a.get(), b.get(), c.get(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    cuda::cuda_mul(a.get(), b.get(), c.get(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
void div(const device_ptr<T[]> &a, const device_ptr<T[]> &b, device_ptr<T[]> &c, size_t size) {
  if (!a.getDevice() || !b.getDevice() || !c.getDevice()) {
    throw std::runtime_error("Device pointer has no associated device");
  }
  if (a.getDevice() != b.getDevice() || a.getDevice() != c.getDevice()) {
    throw std::runtime_error("All device pointers must be on the same device");
  }

  auto device_type = a.getDevice()->getDeviceType();

  if (device_type == DeviceType::CPU) {
    cpu::div(a.get(), b.get(), c.get(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    cuda::cuda_div(a.get(), b.get(), c.get(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

// Fused Multiply-Add Operations

template <typename T>
void fmadd(const device_ptr<T[]> &a, const device_ptr<T[]> &b, device_ptr<T[]> &c, size_t size) {
  if (!a.getDevice() || !b.getDevice() || !c.getDevice()) {
    throw std::runtime_error("Device pointer has no associated device");
  }
  if (a.getDevice() != b.getDevice() || a.getDevice() != c.getDevice()) {
    throw std::runtime_error("All device pointers must be on the same device");
  }

  auto device_type = a.getDevice()->getDeviceType();

  if (device_type == DeviceType::CPU) {
    cpu::fmadd(a.get(), b.get(), c.get(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    cuda::cuda_fmadd(a.get(), b.get(), c.get(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
void fmsub(const device_ptr<T[]> &a, const device_ptr<T[]> &b, device_ptr<T[]> &c, size_t size) {
  if (!a.getDevice() || !b.getDevice() || !c.getDevice()) {
    throw std::runtime_error("Device pointer has no associated device");
  }
  if (a.getDevice() != b.getDevice() || a.getDevice() != c.getDevice()) {
    throw std::runtime_error("All device pointers must be on the same device");
  }

  auto device_type = a.getDevice()->getDeviceType();

  if (device_type == DeviceType::CPU) {
    cpu::fmsub(a.get(), b.get(), c.get(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    cuda::cuda_fmsub(a.get(), b.get(), c.get(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
void fnmadd(const device_ptr<T[]> &a, const device_ptr<T[]> &b, device_ptr<T[]> &c, size_t size) {
  if (!a.getDevice() || !b.getDevice() || !c.getDevice()) {
    throw std::runtime_error("Device pointer has no associated device");
  }
  if (a.getDevice() != b.getDevice() || a.getDevice() != c.getDevice()) {
    throw std::runtime_error("All device pointers must be on the same device");
  }

  auto device_type = a.getDevice()->getDeviceType();

  if (device_type == DeviceType::CPU) {
    cpu::fnmadd(a.get(), b.get(), c.get(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    cuda::cuda_fnmadd(a.get(), b.get(), c.get(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

// Scalar Operations

template <typename T>
void add_scalar(const device_ptr<T[]> &a, T scalar, device_ptr<T[]> &c, size_t size) {
  if (!a.getDevice() || !c.getDevice()) {
    throw std::runtime_error("Device pointer has no associated device");
  }
  if (a.getDevice() != c.getDevice()) {
    throw std::runtime_error("All device pointers must be on the same device");
  }

  auto device_type = a.getDevice()->getDeviceType();

  if (device_type == DeviceType::CPU) {
    cpu::add_scalar(a.get(), scalar, c.get(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    cuda::cuda_add_scalar(a.get(), scalar, c.get(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
void mul_scalar(const device_ptr<T[]> &a, T scalar, device_ptr<T[]> &c, size_t size) {
  if (!a.getDevice() || !c.getDevice()) {
    throw std::runtime_error("Device pointer has no associated device");
  }
  if (a.getDevice() != c.getDevice()) {
    throw std::runtime_error("All device pointers must be on the same device");
  }

  auto device_type = a.getDevice()->getDeviceType();

  if (device_type == DeviceType::CPU) {
    cpu::mul_scalar(a.get(), scalar, c.get(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    cuda::cuda_mul_scalar(a.get(), scalar, c.get(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
void div_scalar(const device_ptr<T[]> &a, T scalar, device_ptr<T[]> &c, size_t size) {
  if (!a.getDevice() || !c.getDevice()) {
    throw std::runtime_error("Device pointer has no associated device");
  }
  if (a.getDevice() != c.getDevice()) {
    throw std::runtime_error("All device pointers must be on the same device");
  }

  auto device_type = a.getDevice()->getDeviceType();

  if (device_type == DeviceType::CPU) {
    cpu::div_scalar(a.get(), scalar, c.get(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    cuda::cuda_div_scalar(a.get(), scalar, c.get(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T> void set_scalar(device_ptr<T[]> &c, T scalar, size_t size) {
  if (!c.getDevice()) {
    throw std::runtime_error("Device pointer has no associated device");
  }

  auto device_type = c.getDevice()->getDeviceType();

  if (device_type == DeviceType::CPU) {
    cpu::set_scalar(c.get(), scalar, size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    cuda::cuda_set_scalar(c.get(), scalar, size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

// Mathematical Functions

template <typename T> void sqrt(const device_ptr<T[]> &a, device_ptr<T[]> &c, size_t size) {
  if (!a.getDevice() || !c.getDevice()) {
    throw std::runtime_error("Device pointer has no associated device");
  }
  if (a.getDevice() != c.getDevice()) {
    throw std::runtime_error("All device pointers must be on the same device");
  }

  auto device_type = a.getDevice()->getDeviceType();

  if (device_type == DeviceType::CPU) {
    cpu::sqrt(a.get(), c.get(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    cuda::cuda_sqrt(a.get(), c.get(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

// Float-only operations
inline void rsqrt(const device_ptr<float[]> &a, device_ptr<float[]> &c, size_t size) {
  if (!a.getDevice() || !c.getDevice()) {
    throw std::runtime_error("Device pointer has no associated device");
  }
  if (a.getDevice() != c.getDevice()) {
    throw std::runtime_error("All device pointers must be on the same device");
  }

  auto device_type = a.getDevice()->getDeviceType();

  if (device_type == DeviceType::CPU) {
    cpu::rsqrt(a.get(), c.get(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    cuda::cuda_rsqrt(a.get(), c.get(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

inline void rcp(const device_ptr<float[]> &a, device_ptr<float[]> &c, size_t size) {
  if (!a.getDevice() || !c.getDevice()) {
    throw std::runtime_error("Device pointer has no associated device");
  }
  if (a.getDevice() != c.getDevice()) {
    throw std::runtime_error("All device pointers must be on the same device");
  }

  auto device_type = a.getDevice()->getDeviceType();

  if (device_type == DeviceType::CPU) {
    cpu::rcp(a.get(), c.get(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    cuda::cuda_rcp(a.get(), c.get(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T> void abs(const device_ptr<T[]> &a, device_ptr<T[]> &c, size_t size) {
  if (!a.getDevice() || !c.getDevice()) {
    throw std::runtime_error("Device pointer has no associated device");
  }
  if (a.getDevice() != c.getDevice()) {
    throw std::runtime_error("All device pointers must be on the same device");
  }

  auto device_type = a.getDevice()->getDeviceType();

  if (device_type == DeviceType::CPU) {
    cpu::abs(a.get(), c.get(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    cuda::cuda_abs(a.get(), c.get(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

// Min/Max Operations

template <typename T>
void min(const device_ptr<T[]> &a, const device_ptr<T[]> &b, device_ptr<T[]> &c, size_t size) {
  if (!a.getDevice() || !b.getDevice() || !c.getDevice()) {
    throw std::runtime_error("Device pointer has no associated device");
  }
  if (a.getDevice() != b.getDevice() || a.getDevice() != c.getDevice()) {
    throw std::runtime_error("All device pointers must be on the same device");
  }

  auto device_type = a.getDevice()->getDeviceType();

  if (device_type == DeviceType::CPU) {
    cpu::min(a.get(), b.get(), c.get(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    cuda::cuda_min(a.get(), b.get(), c.get(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
void max(const device_ptr<T[]> &a, const device_ptr<T[]> &b, device_ptr<T[]> &c, size_t size) {
  if (!a.getDevice() || !b.getDevice() || !c.getDevice()) {
    throw std::runtime_error("Device pointer has no associated device");
  }
  if (a.getDevice() != b.getDevice() || a.getDevice() != c.getDevice()) {
    throw std::runtime_error("All device pointers must be on the same device");
  }

  auto device_type = a.getDevice()->getDeviceType();

  if (device_type == DeviceType::CPU) {
    cpu::max(a.get(), b.get(), c.get(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    cuda::cuda_max(a.get(), b.get(), c.get(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
void scalar_max(const device_ptr<T[]> &a, T scalar, device_ptr<T[]> &c, size_t size) {
  if (!a.getDevice() || !c.getDevice()) {
    throw std::runtime_error("Device pointer has no associated device");
  }
  if (a.getDevice() != c.getDevice()) {
    throw std::runtime_error("All device pointers must be on the same device");
  }

  auto device_type = a.getDevice()->getDeviceType();

  if (device_type == DeviceType::CPU) {
    cpu::scalar_max(a.get(), scalar, c.get(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    cuda::cuda_scalar_max(a.get(), scalar, c.get(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
void clamp(const device_ptr<T[]> &a, T min_val, T max_val, device_ptr<T[]> &c, size_t size) {
  if (!a.getDevice() || !c.getDevice()) {
    throw std::runtime_error("Device pointer has no associated device");
  }
  if (a.getDevice() != c.getDevice()) {
    throw std::runtime_error("All device pointers must be on the same device");
  }

  auto device_type = a.getDevice()->getDeviceType();

  if (device_type == DeviceType::CPU) {
    cpu::clamp(a.get(), min_val, max_val, c.get(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    cuda::cuda_clamp(a.get(), min_val, max_val, c.get(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

// Comparison Operations

template <typename T>
void equal(const device_ptr<T[]> &a, const device_ptr<T[]> &b, device_ptr<T[]> &c, size_t size) {
  if (!a.getDevice() || !b.getDevice() || !c.getDevice()) {
    throw std::runtime_error("Device pointer has no associated device");
  }
  if (a.getDevice() != b.getDevice() || a.getDevice() != c.getDevice()) {
    throw std::runtime_error("All device pointers must be on the same device");
  }

  auto device_type = a.getDevice()->getDeviceType();

  if (device_type == DeviceType::CPU) {
    cpu::equal(a.get(), b.get(), c.get(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    cuda::cuda_equal(a.get(), b.get(), c.get(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
void greater(const device_ptr<T[]> &a, const device_ptr<T[]> &b, device_ptr<T[]> &c, size_t size) {
  if (!a.getDevice() || !b.getDevice() || !c.getDevice()) {
    throw std::runtime_error("Device pointer has no associated device");
  }
  if (a.getDevice() != b.getDevice() || a.getDevice() != c.getDevice()) {
    throw std::runtime_error("All device pointers must be on the same device");
  }

  auto device_type = a.getDevice()->getDeviceType();

  if (device_type == DeviceType::CPU) {
    cpu::greater(a.get(), b.get(), c.get(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    cuda::cuda_greater(a.get(), b.get(), c.get(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

// Memory Operations

template <typename T>
void copy(const device_ptr<T[]> &a, device_ptr<T[]> &c, size_t size, size_t a_offset = 0,
          size_t c_offset = 0) {
  if (!a.getDevice() || !c.getDevice()) {
    throw std::runtime_error("Device pointer has no associated device");
  }
  if (a.getDevice() != c.getDevice()) {
    throw std::runtime_error("All device pointers must be on the same device");
  }

  if (a.get() == nullptr || c.get() == nullptr) {
    throw std::runtime_error("Null pointer exception in copy operation");
  }

  auto device_type = a.getDevice()->getDeviceType();

  if (device_type == DeviceType::CPU) {
    cpu::copy(a.get() + a_offset, c.get() + c_offset, size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    cuda::cuda_copy(a.get() + a_offset, c.get() + c_offset, size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T> void zero(device_ptr<T[]> &c, size_t size) {
  if (!c.getDevice()) {
    throw std::runtime_error("Device pointer has no associated device");
  }

  auto device_type = c.getDevice()->getDeviceType();

  if (device_type == DeviceType::CPU) {
    cpu::zero(c.get(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    cuda::cuda_zero(c.get(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

// Reduction Operations

template <typename T> T sum(const device_ptr<T[]> &a, size_t size) {
  if (!a.getDevice()) {
    throw std::runtime_error("Device pointer has no associated device");
  }

  auto device_type = a.getDevice()->getDeviceType();

  if (device_type == DeviceType::CPU) {
    return cpu::sum(a.get(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return cuda::cuda_sum(a.get(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
T dot_product(const device_ptr<T[]> &a, const device_ptr<T[]> &b, size_t size) {
  if (!a.getDevice() || !b.getDevice()) {
    throw std::runtime_error("Device pointer has no associated device");
  }
  if (a.getDevice() != b.getDevice()) {
    throw std::runtime_error("All device pointers must be on the same device");
  }

  auto device_type = a.getDevice()->getDeviceType();

  if (device_type == DeviceType::CPU) {
    return cpu::dot_product(a.get(), b.get(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return cuda::cuda_dot_product(a.get(), b.get(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T> T norm_squared(const device_ptr<T[]> &a, size_t size) {
  if (!a.getDevice()) {
    throw std::runtime_error("Device pointer has no associated device");
  }

  auto device_type = a.getDevice()->getDeviceType();

  if (device_type == DeviceType::CPU) {
    return cpu::norm_squared(a.get(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return cuda::cuda_norm_squared(a.get(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T> T sum_squared_diff(const device_ptr<T[]> &a, T mean, size_t size) {
  if (!a.getDevice()) {
    throw std::runtime_error("Device pointer has no associated device");
  }

  auto device_type = a.getDevice()->getDeviceType();

  if (device_type == DeviceType::CPU) {
    return cpu::sum_squared_diff(a.get(), mean, size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return cuda::cuda_sum_squared_diff(a.get(), mean, size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

// Specialized BatchNorm Operations

template <typename T>
void sub_mul_scalar(const device_ptr<T[]> &a, T sub_scalar, T mul_scalar, device_ptr<T[]> &c,
                    size_t size) {
  if (!a.getDevice() || !c.getDevice()) {
    throw std::runtime_error("Device pointer has no associated device");
  }
  if (a.getDevice() != c.getDevice()) {
    throw std::runtime_error("All device pointers must be on the same device");
  }

  auto device_type = a.getDevice()->getDeviceType();

  if (device_type == DeviceType::CPU) {
    cpu::sub_mul_scalar(a.get(), sub_scalar, mul_scalar, c.get(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    cuda::cuda_sub_mul_scalar(a.get(), sub_scalar, mul_scalar, c.get(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
void mul_add_scalar(const device_ptr<T[]> &a, T mul_scalar, T add_scalar, device_ptr<T[]> &c,
                    size_t size) {
  if (!a.getDevice() || !c.getDevice()) {
    throw std::runtime_error("Device pointer has no associated device");
  }
  if (a.getDevice() != c.getDevice()) {
    throw std::runtime_error("All device pointers must be on the same device");
  }

  auto device_type = a.getDevice()->getDeviceType();

  if (device_type == DeviceType::CPU) {
    cpu::mul_add_scalar(a.get(), mul_scalar, add_scalar, c.get(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    cuda::cuda_mul_add_scalar(a.get(), mul_scalar, add_scalar, c.get(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

// Random number generation operations

template <typename T>
void fill_random_uniform(device_ptr<T[]> &data, size_t size, T min_val, T max_val,
                         unsigned long long seed) {
  if (!data.getDevice()) {
    throw std::runtime_error("Device pointer has no associated device");
  }

  auto device_type = data.getDevice()->getDeviceType();

  if (device_type == DeviceType::CPU) {
    cpu::fill_random_uniform(data.get(), size, min_val, max_val, seed);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    cuda::cuda_fill_random_uniform(data.get(), size, min_val, max_val, seed);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
void fill_random_normal(device_ptr<T[]> &data, size_t size, T mean, T stddev,
                        unsigned long long seed) {
  if (!data.getDevice()) {
    throw std::runtime_error("Device pointer has no associated device");
  }

  auto device_type = data.getDevice()->getDeviceType();

  if (device_type == DeviceType::CPU) {
    cpu::fill_random_normal(data.get(), size, mean, stddev, seed);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    cuda::cuda_fill_random_normal(data.get(), size, mean, stddev, seed);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
void transpose_2d(const device_ptr<T[]> &input, device_ptr<T[]> &output, size_t rows, size_t cols) {
  if (!input.getDevice() || !output.getDevice()) {
    throw std::runtime_error("Device pointer has no associated device");
  }

  if (output.getDevice() != input.getDevice()) {
    throw std::runtime_error("Input and output must be on the same device");
  }

  auto device_type = input.getDevice()->getDeviceType();

  if (device_type == DeviceType::CPU) {
    cpu::transpose_2d(input.get(), output.get(), rows, cols);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    cuda::cuda_transpose_2d(input.get(), output.get(), rows, cols);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
void nchw_to_cnhw(const device_ptr<T[]> &input, device_ptr<T[]> &output, size_t n, size_t c,
                  size_t h, size_t w) {
  if (!input.getDevice() || !output.getDevice()) {
    throw std::runtime_error("Device pointer has no associated device");
  }

  if (output.getDevice() != input.getDevice()) {
    throw std::runtime_error("Input and output must be on the same device");
  }

  auto device_type = input.getDevice()->getDeviceType();

  if (device_type == DeviceType::CPU) {
    cpu::nchw_to_cnhw(input.get(), output.get(), n, c, h, w);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    cuda::cuda_nchw_to_cnhw(input.get(), output.get(), n, c, h, w);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
void cnhw_to_nchw(const device_ptr<T[]> &input, device_ptr<T[]> &output, size_t n, size_t c,
                  size_t h, size_t w) {
  if (!input.getDevice() || !output.getDevice()) {
    throw std::runtime_error("Device pointer has no associated device");
  }
  if (output.getDevice() != input.getDevice()) {
    throw std::runtime_error("Input and output must be on the same device");
  }
  auto device_type = input.getDevice()->getDeviceType();
  if (device_type == DeviceType::CPU) {
    cpu::cnhw_to_nchw(input.get(), output.get(), n, c, h, w);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    cuda::cuda_cnhw_to_nchw(input.get(), output.get(), n, c, h, w);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

} // namespace ops
} // namespace tnn