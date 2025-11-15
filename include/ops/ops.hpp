#pragma once

#include "cpu/kernels.hpp"
#include "device/device_ptr.hpp"
#ifdef USE_CUDA
#include "cuda/kernels.hpp"
#endif
#include "device/async_context.hpp"

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>

namespace tnn {
namespace ops {

template <typename CpuKernelFunc, typename T, typename... Args
#ifdef USE_CUDA
          ,
          typename CudaKernelFunc
#endif
          >
std::unique_ptr<Task> dispatch_op(CpuKernelFunc cpu_kernel,
#ifdef USE_CUDA
                                  CudaKernelFunc cuda_kernel,
#endif
                                  const device_ptr<T[]> &a, Args... args) {

  if (!a.getDevice()) {
    throw std::runtime_error("Device pointer has no associated device");
  }

  auto device = a.getDevice();
  auto device_type = device->getDeviceType();

  if (device_type == DeviceType::CPU) {

    return std::make_unique<CPUTask>(cpu_kernel, device, a.get(), std::forward<Args>(args)...);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {

    return std::make_unique<CUDATask>(cuda_kernel, device, a.get(), std::forward<Args>(args)...);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> add(const device_ptr<T[]> &a, const device_ptr<T[]> &b, device_ptr<T[]> &c,
                          size_t size) {
  if (a.getDevice() != b.getDevice() || a.getDevice() != c.getDevice()) {
    throw std::runtime_error("All device pointers must be on the same device");
  }

  using CudaAddKernel = void (*)(const T *, const T *, T *, size_t, cudaStream_t);

  return dispatch_op((void (*)(const T *, const T *, T *, size_t))cpu::add,
#ifdef USE_CUDA
                     static_cast<CudaAddKernel>(cuda::cuda_add),
#endif
                     a, b.get(), c.get(), size);
}

template <typename T>
std::unique_ptr<Task> sub(const device_ptr<T[]> &a, const device_ptr<T[]> &b, device_ptr<T[]> &c,
                          size_t size) {
  if (a.getDevice() != b.getDevice() || a.getDevice() != c.getDevice()) {
    throw std::runtime_error("All device pointers must be on the same device");
  }
  using CudaSubKernel = void (*)(const T *, const T *, T *, size_t, cudaStream_t);

  return dispatch_op((void (*)(const T *, const T *, T *, size_t))cpu::sub,
#ifdef USE_CUDA
                     static_cast<CudaSubKernel>(cuda::cuda_sub),
#endif
                     a, b.get(), c.get(), size);
}

template <typename T>
std::unique_ptr<Task> mul(const device_ptr<T[]> &a, const device_ptr<T[]> &b, device_ptr<T[]> &c,
                          size_t size) {
  if (a.getDevice() != b.getDevice() || a.getDevice() != c.getDevice()) {
    throw std::runtime_error("All device pointers must be on the same device");
  }
  using CudaMulKernel = void (*)(const T *, const T *, T *, size_t, cudaStream_t);

  return dispatch_op((void (*)(const T *, const T *, T *, size_t))cpu::mul,
#ifdef USE_CUDA
                     static_cast<CudaMulKernel>(cuda::cuda_mul),
#endif
                     a, b.get(), c.get(), size);
}

template <typename T>
std::unique_ptr<Task> div(const device_ptr<T[]> &a, const device_ptr<T[]> &b, device_ptr<T[]> &c,
                          size_t size) {
  if (a.getDevice() != b.getDevice() || a.getDevice() != c.getDevice()) {
    throw std::runtime_error("All device pointers must be on the same device");
  }
  using CudaDivKernel = void (*)(const T *, const T *, T *, size_t, cudaStream_t);

  return dispatch_op((void (*)(const T *, const T *, T *, size_t))cpu::div,
#ifdef USE_CUDA
                     static_cast<CudaDivKernel>(cuda::cuda_div),
#endif
                     a, b.get(), c.get(), size);
}

template <typename T>
std::unique_ptr<Task> fmadd(const device_ptr<T[]> &a, const device_ptr<T[]> &b, device_ptr<T[]> &c,
                            size_t size) {
  if (a.getDevice() != b.getDevice() || a.getDevice() != c.getDevice()) {
    throw std::runtime_error("All device pointers must be on the same device");
  }
  using CudaFMADDKernel = void (*)(const T *, const T *, T *, size_t, cudaStream_t);

  return dispatch_op((void (*)(const T *, const T *, T *, size_t))cpu::fmadd,
#ifdef USE_CUDA
                     static_cast<CudaFMADDKernel>(cuda::cuda_fmadd),
#endif
                     a, b.get(), c.get(), size);
}

template <typename T>
std::unique_ptr<Task> fmsub(const device_ptr<T[]> &a, const device_ptr<T[]> &b, device_ptr<T[]> &c,
                            size_t size) {
  if (a.getDevice() != b.getDevice() || a.getDevice() != c.getDevice()) {
    throw std::runtime_error("All device pointers must be on the same device");
  }
  using CudaFMSUBKernel = void (*)(const T *, const T *, T *, size_t, cudaStream_t);

  return dispatch_op((void (*)(const T *, const T *, T *, size_t))cpu::fmsub,
#ifdef USE_CUDA
                     static_cast<CudaFMSUBKernel>(cuda::cuda_fmsub),
#endif
                     a, b.get(), c.get(), size);
}

template <typename T>
std::unique_ptr<Task> fnmadd(const device_ptr<T[]> &a, const device_ptr<T[]> &b, device_ptr<T[]> &c,
                             size_t size) {
  if (a.getDevice() != b.getDevice() || a.getDevice() != c.getDevice()) {
    throw std::runtime_error("All device pointers must be on the same device");
  }
  using CudaFNMADDKernel = void (*)(const T *, const T *, T *, size_t, cudaStream_t);

  return dispatch_op((void (*)(const T *, const T *, T *, size_t))cpu::fnmadd,
#ifdef USE_CUDA
                     static_cast<CudaFNMADDKernel>(cuda::cuda_fnmadd),
#endif
                     a, b.get(), c.get(), size);
}

template <typename T>
std::unique_ptr<Task> add_scalar(const device_ptr<T[]> &a, T scalar, device_ptr<T[]> &c,
                                 size_t size) {
  if (a.getDevice() != c.getDevice()) {
    throw std::runtime_error("All device pointers must be on the same device");
  }
  using CudaAddScalarKernel = void (*)(const T *, T, T *, size_t, cudaStream_t);

  return dispatch_op((void (*)(const T *, T, T *, size_t))cpu::add_scalar,
#ifdef USE_CUDA
                     static_cast<CudaAddScalarKernel>(cuda::cuda_add_scalar),
#endif
                     a, scalar, c.get(), size);
}

template <typename T>
std::unique_ptr<Task> mul_scalar(const device_ptr<T[]> &a, T scalar, device_ptr<T[]> &c,
                                 size_t size) {
  if (a.getDevice() != c.getDevice()) {
    throw std::runtime_error("All device pointers must be on the same device");
  }
  using CudaMulScalarKernel = void (*)(const T *, T, T *, size_t, cudaStream_t);

  return dispatch_op((void (*)(const T *, T, T *, size_t))cpu::mul_scalar,
#ifdef USE_CUDA
                     static_cast<CudaMulScalarKernel>(cuda::cuda_mul_scalar),
#endif
                     a, scalar, c.get(), size);
}

template <typename T>
std::unique_ptr<Task> div_scalar(const device_ptr<T[]> &a, T scalar, device_ptr<T[]> &c,
                                 size_t size) {
  if (a.getDevice() != c.getDevice()) {
    throw std::runtime_error("All device pointers must be on the same device");
  }
  using CudaDivScalarKernel = void (*)(const T *, T, T *, size_t, cudaStream_t);

  return dispatch_op((void (*)(const T *, T, T *, size_t))cpu::div_scalar,
#ifdef USE_CUDA
                     static_cast<CudaDivScalarKernel>(cuda::cuda_div_scalar),
#endif
                     a, scalar, c.get(), size);
}

template <typename T> std::unique_ptr<Task> set_scalar(device_ptr<T[]> &c, T scalar, size_t size) {
  if (!c.getDevice()) {
    throw std::runtime_error("Device pointer has no associated device");
  }
  auto device = c.getDevice();
  auto device_type = device->getDeviceType();

  if (device_type == DeviceType::CPU) {
    return std::make_unique<CPUTask>((void (*)(T *, T, size_t))cpu::set_scalar, device, c.get(),
                                     scalar, size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    using CudaSetScalarKernel = void (*)(T *, T, size_t, cudaStream_t);
    return std::make_unique<CUDATask>(static_cast<CudaSetScalarKernel>(cuda::cuda_set_scalar),
                                      device, c.get(), scalar, size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> sqrt(const device_ptr<T[]> &a, device_ptr<T[]> &c, size_t size) {
  if (a.getDevice() != c.getDevice()) {
    throw std::runtime_error("All device pointers must be on the same device");
  }
  using CudaSqrtKernel = void (*)(const T *, T *, size_t, cudaStream_t);

  return dispatch_op((void (*)(const T *, T *, size_t))cpu::sqrt,
#ifdef USE_CUDA
                     static_cast<CudaSqrtKernel>(cuda::cuda_sqrt),
#endif
                     a, c.get(), size);
}

inline std::unique_ptr<Task> rsqrt(const device_ptr<float[]> &a, device_ptr<float[]> &c,
                                   size_t size) {
  if (a.getDevice() != c.getDevice()) {
    throw std::runtime_error("All device pointers must be on the same device");
  }
  using CudaRSqrtKernel = void (*)(const float *, float *, size_t, cudaStream_t);

  return dispatch_op((void (*)(const float *, float *, size_t))cpu::rsqrt,
#ifdef USE_CUDA
                     static_cast<CudaRSqrtKernel>(cuda::cuda_rsqrt),
#endif
                     a, c.get(), size);
}

inline std::unique_ptr<Task> rcp(const device_ptr<float[]> &a, device_ptr<float[]> &c,
                                 size_t size) {
  if (a.getDevice() != c.getDevice()) {
    throw std::runtime_error("All device pointers must be on the same device");
  }
  using CudaRCPKernel = void (*)(const float *, float *, size_t, cudaStream_t);

  return dispatch_op((void (*)(const float *, float *, size_t))cpu::rcp,
#ifdef USE_CUDA
                     static_cast<CudaRCPKernel>(cuda::cuda_rcp),
#endif
                     a, c.get(), size);
}

template <typename T>
std::unique_ptr<Task> abs(const device_ptr<T[]> &a, device_ptr<T[]> &c, size_t size) {
  if (a.getDevice() != c.getDevice()) {
    throw std::runtime_error("All device pointers must be on the same device");
  }
  using CudaAbsKernel = void (*)(const T *, T *, size_t, cudaStream_t);

  return dispatch_op((void (*)(const T *, T *, size_t))cpu::abs,
#ifdef USE_CUDA
                     static_cast<CudaAbsKernel>(cuda::cuda_abs),
#endif
                     a, c.get(), size);
}

template <typename T>
std::unique_ptr<Task> min(const device_ptr<T[]> &a, const device_ptr<T[]> &b, device_ptr<T[]> &c,
                          size_t size) {
  if (a.getDevice() != b.getDevice() || a.getDevice() != c.getDevice()) {
    throw std::runtime_error("All device pointers must be on the same device");
  }
  using CudaMinKernel = void (*)(const T *, const T *, T *, size_t, cudaStream_t);

  return dispatch_op((void (*)(const T *, const T *, T *, size_t))cpu::min,
#ifdef USE_CUDA
                     static_cast<CudaMinKernel>(cuda::cuda_min),
#endif
                     a, b.get(), c.get(), size);
}

template <typename T>
std::unique_ptr<Task> max(const device_ptr<T[]> &a, const device_ptr<T[]> &b, device_ptr<T[]> &c,
                          size_t size) {
  if (a.getDevice() != b.getDevice() || a.getDevice() != c.getDevice()) {
    throw std::runtime_error("All device pointers must be on the same device");
  }
  using CudaMaxKernel = void (*)(const T *, const T *, T *, size_t, cudaStream_t);

  return dispatch_op((void (*)(const T *, const T *, T *, size_t))cpu::max,
#ifdef USE_CUDA
                     static_cast<CudaMaxKernel>(cuda::cuda_max),
#endif
                     a, b.get(), c.get(), size);
}

template <typename T>
std::unique_ptr<Task> scalar_max(const device_ptr<T[]> &a, T scalar, device_ptr<T[]> &c,
                                 size_t size) {
  if (a.getDevice() != c.getDevice()) {
    throw std::runtime_error("All device pointers must be on the same device");
  }
  using CudaScalarMaxKernel = void (*)(const T *, T, T *, size_t, cudaStream_t);

  return dispatch_op((void (*)(const T *, T, T *, size_t))cpu::scalar_max,
#ifdef USE_CUDA
                     static_cast<CudaScalarMaxKernel>(cuda::cuda_scalar_max),
#endif
                     a, scalar, c.get(), size);
}

template <typename T>
std::unique_ptr<Task> clamp(const device_ptr<T[]> &a, T min_val, T max_val, device_ptr<T[]> &c,
                            size_t size) {
  if (a.getDevice() != c.getDevice()) {
    throw std::runtime_error("All device pointers must be on the same device");
  }
  using CudaClampKernel = void (*)(const T *, T, T, T *, size_t, cudaStream_t);

  return dispatch_op((void (*)(const T *, T, T, T *, size_t))cpu::clamp,
#ifdef USE_CUDA
                     static_cast<CudaClampKernel>(cuda::cuda_clamp),
#endif
                     a, min_val, max_val, c.get(), size);
}

template <typename T>
std::unique_ptr<Task> equal(const device_ptr<T[]> &a, const device_ptr<T[]> &b, device_ptr<T[]> &c,
                            size_t size) {
  if (a.getDevice() != b.getDevice() || a.getDevice() != c.getDevice()) {
    throw std::runtime_error("All device pointers must be on the same device");
  }
  using CudaEqualKernel = void (*)(const T *, const T *, T *, size_t, cudaStream_t);

  return dispatch_op((void (*)(const T *, const T *, T *, size_t))cpu::equal,
#ifdef USE_CUDA
                     static_cast<CudaEqualKernel>(cuda::cuda_equal),
#endif
                     a, b.get(), c.get(), size);
}

template <typename T>
std::unique_ptr<Task> greater(const device_ptr<T[]> &a, const device_ptr<T[]> &b,
                              device_ptr<T[]> &c, size_t size) {
  if (a.getDevice() != b.getDevice() || a.getDevice() != c.getDevice()) {
    throw std::runtime_error("All device pointers must be on the same device");
  }
  using CudaGreaterKernel = void (*)(const T *, const T *, T *, size_t, cudaStream_t);

  return dispatch_op((void (*)(const T *, const T *, T *, size_t))cpu::greater,
#ifdef USE_CUDA
                     static_cast<CudaGreaterKernel>(cuda::cuda_greater),
#endif
                     a, b.get(), c.get(), size);
}

template <typename T>
std::unique_ptr<Task> copy(const device_ptr<T[]> &a, device_ptr<T[]> &c, size_t size,
                           size_t a_offset = 0, size_t c_offset = 0) {
  if (a.getDevice() != c.getDevice()) {
    throw std::runtime_error("All device pointers must be on the same device");
  }
  if (a.get() == nullptr || c.get() == nullptr) {
    throw std::runtime_error("Null pointer exception in copy operation");
  }

  auto cpu_kernel = [](const T *a_ptr, T *c_ptr, size_t s) { cpu::copy(a_ptr, c_ptr, s); };

#ifdef USE_CUDA

  using CudaCopyKernel = void (*)(const T *, T *, size_t, cudaStream_t);

  auto cuda_kernel = static_cast<CudaCopyKernel>(cuda::cuda_copy);
#endif

  auto device = a.getDevice();
  auto device_type = device->getDeviceType();

  if (device_type == DeviceType::CPU) {
    return std::make_unique<CPUTask>(cpu_kernel, device, a.get() + a_offset, c.get() + c_offset,
                                     size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return std::make_unique<CUDATask>(cuda_kernel, device, a.get() + a_offset, c.get() + c_offset,
                                      size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T> std::unique_ptr<Task> zero(device_ptr<T[]> &c, size_t size) {
  if (!c.getDevice()) {
    throw std::runtime_error("Device pointer has no associated device");
  }
  auto device = c.getDevice();
  auto device_type = device->getDeviceType();

  if (device_type == DeviceType::CPU) {
    return std::make_unique<CPUTask>((void (*)(T *, size_t))cpu::zero, device, c.get(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    using CudaZeroKernel = void (*)(T *, size_t, cudaStream_t);
    return std::make_unique<CUDATask>(static_cast<CudaZeroKernel>(cuda::cuda_zero), device, c.get(),
                                      size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

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
    return cuda::cuda_sum(a.get(), size, 0);
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
    return cuda::cuda_sum_squared_diff(a.get(), mean, size, 0);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> sub_mul_scalar(const device_ptr<T[]> &a, T sub_scalar, T mul_scalar,
                                     device_ptr<T[]> &c, size_t size) {
  if (a.getDevice() != c.getDevice()) {
    throw std::runtime_error("All device pointers must be on the same device");
  }
  using CudaSubMulScalarKernel = void (*)(const T *, T, T, T *, size_t, cudaStream_t);

  return dispatch_op((void (*)(const T *, T, T, T *, size_t))cpu::sub_mul_scalar,
#ifdef USE_CUDA
                     static_cast<CudaSubMulScalarKernel>(cuda::cuda_sub_mul_scalar),
#endif
                     a, sub_scalar, mul_scalar, c.get(), size);
}

template <typename T>
std::unique_ptr<Task> mul_add_scalar(const device_ptr<T[]> &a, T mul_scalar, T add_scalar,
                                     device_ptr<T[]> &c, size_t size) {
  if (a.getDevice() != c.getDevice()) {
    throw std::runtime_error("All device pointers must be on the same device");
  }
  using CudaMulAddScalarKernel = void (*)(const T *, T, T, T *, size_t, cudaStream_t);

  return dispatch_op((void (*)(const T *, T, T, T *, size_t))cpu::mul_add_scalar,
#ifdef USE_CUDA
                     static_cast<CudaMulAddScalarKernel>(cuda::cuda_mul_add_scalar),
#endif
                     a, mul_scalar, add_scalar, c.get(), size);
}

template <typename T>
std::unique_ptr<Task> fill_random_uniform(device_ptr<T[]> &data, size_t size, T min_val, T max_val,
                                          unsigned long long seed) {
  if (!data.getDevice()) {
    throw std::runtime_error("Device pointer has no associated device");
  }
  auto device = data.getDevice();
  auto device_type = device->getDeviceType();

  if (device_type == DeviceType::CPU) {
    return std::make_unique<CPUTask>(
        (void (*)(T *, size_t, T, T, unsigned long long))cpu::fill_random_uniform, device,
        data.get(), size, min_val, max_val, seed);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    using CudaRandomUniformKernel = void (*)(T *, size_t, T, T, unsigned long long, cudaStream_t);
    return std::make_unique<CUDATask>(
        static_cast<CudaRandomUniformKernel>(cuda::cuda_fill_random_uniform), device, data.get(),
        size, min_val, max_val, seed);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> fill_random_normal(device_ptr<T[]> &data, size_t size, T mean, T stddev,
                                         unsigned long long seed) {
  if (!data.getDevice()) {
    throw std::runtime_error("Device pointer has no associated device");
  }
  auto device = data.getDevice();
  auto device_type = device->getDeviceType();

  if (device_type == DeviceType::CPU) {
    return std::make_unique<CPUTask>(
        (void (*)(T *, size_t, T, T, unsigned long long))cpu::fill_random_normal, device,
        data.get(), size, mean, stddev, seed);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    using CudaRandomNormalKernel = void (*)(T *, size_t, T, T, unsigned long long, cudaStream_t);
    return std::make_unique<CUDATask>(
        static_cast<CudaRandomNormalKernel>(cuda::cuda_fill_random_normal), device, data.get(),
        size, mean, stddev, seed);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> transpose_2d(const device_ptr<T[]> &input, device_ptr<T[]> &output,
                                   size_t rows, size_t cols) {
  if (output.getDevice() != input.getDevice()) {
    throw std::runtime_error("Input and output must be on the same device");
  }
  using CudaTransposeKernel = void (*)(const T *, T *, size_t, size_t, cudaStream_t);

  return dispatch_op((void (*)(const T *, T *, size_t, size_t))cpu::transpose_2d,
#ifdef USE_CUDA
                     static_cast<CudaTransposeKernel>(cuda::cuda_transpose_2d),
#endif
                     input, output.get(), rows, cols);
}

template <typename T>
std::unique_ptr<Task> nchw_to_cnhw(const device_ptr<T[]> &input, device_ptr<T[]> &output, size_t n,
                                   size_t c, size_t h, size_t w) {
  if (output.getDevice() != input.getDevice()) {
    throw std::runtime_error("Input and output must be on the same device");
  }
  using CudaNCHWToCNHWKernel =
      void (*)(const T *, T *, size_t, size_t, size_t, size_t, cudaStream_t);

  return dispatch_op((void (*)(const T *, T *, size_t, size_t, size_t, size_t))cpu::nchw_to_cnhw,
#ifdef USE_CUDA
                     static_cast<CudaNCHWToCNHWKernel>(cuda::cuda_nchw_to_cnhw),
#endif
                     input, output.get(), n, c, h, w);
}

template <typename T>
std::unique_ptr<Task> cnhw_to_nchw(const device_ptr<T[]> &input, device_ptr<T[]> &output, size_t n,
                                   size_t c, size_t h, size_t w) {
  if (output.getDevice() != input.getDevice()) {
    throw std::runtime_error("Input and output must be on the same device");
  }
  using CudaCNHWToNCHWKernel =
      void (*)(const T *, T *, size_t, size_t, size_t, size_t, cudaStream_t);

  return dispatch_op((void (*)(const T *, T *, size_t, size_t, size_t, size_t))cpu::cnhw_to_nchw,
#ifdef USE_CUDA
                     static_cast<CudaCNHWToNCHWKernel>(cuda::cuda_cnhw_to_nchw),
#endif
                     input, output.get(), n, c, h, w);
}

} // namespace ops
} // namespace tnn