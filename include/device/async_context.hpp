#pragma once

#include "device/device.hpp"
#include "device/device_ptr.hpp"
#include <functional>
#include <thread>

namespace tnn {

template <typename TRType> class AsyncContext {
public:
  virtual device_ptr<TRType> synchronize() const = 0;
};

template <typename TRType> class CPUAsyncContext : public AsyncContext<TRType> {
private:
  std::function<TRType()> func_;
  device_ptr<TRType> result_;
  std::thread thread_;
  const Device *device_;

public:
  explicit CPUAsyncContext(std::function<TRType()> func, const Device *device)
      : func_(func), device_(device) {
    result_ = make_ptr<TRType>(device_);
    thread_ = std::thread([this]() { *(result_.get()) = func_(); });
  }

  device_ptr<TRType> synchronize() override {
    if (thread_.joinable()) {
      thread_.join();
    }
    return result_;
  }
};

#ifdef USE_CUDA

#include <cuda_runtime.h>

template <typename TRType> class CUDAAsyncContext : public AsyncContext<TRType> {
private:
  device_ptr<TRType> result_ptr_;
  cudaStream_t stream_;
  cudaEvent_t event_;

  std::function<void(TRType *, cudaStream_t)> launch_func_;

  const Device *device_;

public:
  explicit CUDAAsyncContext(std::function<void(TRType *, cudaStream_t)> func, cudaStream_t stream,
                            const Device *device)
      : launch_func_(func), stream_(stream), device_(device) {
    // allocate memory, create event, launch kernel, record event
    result_ptr_ = make_ptr<TRType>(device_);

    cudaEventCreate(&event_);

    launch_func_(result_ptr_.get(), stream_);

    cudaEventRecord(event_, stream_);
  }

  ~CUDAAsyncContext() { cudaEventDestroy(event_); }

  device_ptr<TRType> synchronize() const override {
    cudaEventSynchronize(event_);

    return result_ptr_;
  }

  // delete for now, since we don't want to deal with very unique resources
  CUDAAsyncContext(const CUDAAsyncContext &) = delete;
  CUDAAsyncContext &operator=(const CUDAAsyncContext &) = delete;
  CUDAAsyncContext(CUDAAsyncContext &&) = delete;
  CUDAAsyncContext &operator=(CUDAAsyncContext &&) = delete;
};
#endif // USE_CUDA

} // namespace tnn
