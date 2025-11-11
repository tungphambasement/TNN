#pragma once

#include "device/device.hpp"
#include "device/device_ptr.hpp"
#include <functional>
#include <thread>
#include <type_traits>
#include <utility> // For std::forward

namespace tnn {
template <typename TRType> class AsyncContext {
public:
  virtual device_ptr<TRType> synchronize() = 0;
  virtual ~AsyncContext() = default;
};

template <typename TRType> class CPUAsyncContext : public AsyncContext<TRType> {
private:
  std::function<void()> func_;
  std::thread thread_;
  const Device *device_;

public:
  template <typename Func, typename... Args>
  explicit CPUAsyncContext(Func &&func, const Device *device, Args &&...args) : device_(device) {
    func_ = [f = std::forward<Func>(func), ... a = std::forward<Args>(args)]() mutable -> TRType {
      return f(std::forward<Args>(a)...);
    };

    thread_ = std::thread([this]() { func_(); });
  }

  ~CPUAsyncContext() {
    if (thread_.joinable()) {
      thread_.join();
    }
  }

  void synchronize() override {
    if (thread_.joinable()) {
      thread_.join();
    }
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
  cudaStream_t stream_;
  cudaEvent_t event_;

  std::function<void()> launch_func_;

  const Device *device_;

public:
  template <typename Func, typename... Args>
  explicit CUDAAsyncContext(Func &&func, const Device *device, Args &&...args) : device_(device) {

    cudaStreamCreate(&stream_);

    auto check_cuda = [](cudaError_t err, const char *msg) {
      if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
      }
    };

    check_cuda(cudaEventCreate(&event_), "cudaEventCreate failed");

    launch_func_ = [f = std::forward<Func>(func), ... a = std::forward<Args>(args),
                    stream = stream_]() mutable { f(std::forward<Args>(a)..., stream); };

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

void wait_all(std::vector<std::unique_ptr<AsyncContext<void>>> &contexts) {
  for (auto &ctx : contexts) {
    ctx->synchronize();
  }
}
} // namespace tnn