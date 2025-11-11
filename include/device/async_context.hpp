#pragma once

#include "device/device.hpp"
#include "device/device_ptr.hpp"
#include <functional>
#include <thread>
#include <utility> // For std::forward

namespace tnn {

template <typename TRType> class AsyncContext {
public:
  virtual device_ptr<TRType> synchronize() = 0;
  virtual ~AsyncContext() = default;
};

template <typename TRType> class CPUAsyncContext : public AsyncContext<TRType> {
private:
  std::function<TRType()> func_;
  device_ptr<TRType> result_;
  std::thread thread_;
  const Device *device_;

public:
  template <typename Func, typename... Args>
  explicit CPUAsyncContext(Func &&func, const Device *device, Args &&...args) : device_(device) {
    func_ = [f = std::forward<Func>(func), ... a = std::forward<Args>(args)]() mutable -> TRType {
      return f(std::forward<Args>(a)...);
    };

    result_ = make_ptr<TRType>(device_);
    thread_ = std::thread([this]() { *(result_.get()) = func_(); });
  }

  ~CPUAsyncContext() {
    if (thread_.joinable()) {
      thread_.join();
    }
  }

  device_ptr<TRType> synchronize() override {
    if (thread_.joinable()) {
      thread_.join();
    }
    return result_;
  }

  CPUAsyncContext(const CPUAsyncContext &) = delete;
  CPUAsyncContext &operator=(const CPUAsyncContext &) = delete;
  CPUAsyncContext(CPUAsyncContext &&) = delete;
  CPUAsyncContext &operator=(CPUAsyncContext &&) = delete;
};

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <stdexcept>

template <typename TRType> class CUDAAsyncContext : public AsyncContext<TRType> {
private:
  device_ptr<TRType> result_ptr_;
  cudaStream_t stream_;
  cudaEvent_t event_;

  std::function<void()> launch_func_;

  const Device *device_;

public:
  template <typename Func, typename... Args>
  explicit CUDAAsyncContext(Func &&func, const Device *device, Args &&...args) : device_(device) {
    result_ptr_ = make_ptr<TRType>(device_);

    cudaStreamCreate(&stream_);

    auto check_cuda = [](cudaError_t err, const char *msg) {
      if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
      }
    };

    check_cuda(cudaEventCreate(&event_), "cudaEventCreate failed");

    launch_func_ = [f = std::forward<Func>(func), ... a = std::forward<Args>(args),
                    result_ptr = result_ptr_.get(), stream = stream_]() mutable {
      f(std::forward<Args>(a)..., result_ptr, stream);
    };

    launch_func_();

    check_cuda(cudaEventRecord(event_, stream_), "cudaEventRecord failed");
  }

  ~CUDAAsyncContext() { cudaEventDestroy(event_); }

  device_ptr<TRType> synchronize() override {
    cudaEventSynchronize(event_);

    return result_ptr_;
  }

  CUDAAsyncContext(const CUDAAsyncContext &) = delete;
  CUDAAsyncContext &operator=(const CUDAAsyncContext &) = delete;
  CUDAAsyncContext(CUDAAsyncContext &&) = delete;
  CUDAAsyncContext &operator=(CUDAAsyncContext &&) = delete;
};

#endif // USE_CUDA

} // namespace tnn