#pragma once

#include "device/device.hpp" // Assuming this is defined elsewhere
#include "threadpool.hpp"

#include <future>
#include <memory>
#include <system_error>
#include <tuple>
#include <utility>

namespace tnn {

using ErrorStatus = std::error_code;

class Task {
public:
  virtual ~Task() = default;
  virtual ErrorStatus synchronize() = 0;
};

class CPUTask : public Task {
private:
  const Device *device_;

public:
  template <typename Func, typename... Args>
  explicit CPUTask(Func &&func, const Device *device, Args &&...args) : device_(device) {
    auto bound_work = [f = std::forward<Func>(func),
                       args_tuple = std::make_tuple(std::forward<Args>(args)...)]() mutable {
      std::apply(f, std::move(args_tuple));
    };

    bound_work();
  }

  ~CPUTask() {}

  ErrorStatus synchronize() override { return ErrorStatus{}; }

  CPUTask(const CPUTask &) = delete;
  CPUTask &operator=(const CPUTask &) = delete;
  CPUTask(CPUTask &&) = delete;
  CPUTask &operator=(CPUTask &&) = delete;
};

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <stdexcept>

inline ErrorStatus cuda_error_to_status(cudaError_t err) {
  if (err == cudaSuccess) {
    return ErrorStatus{};
  }
  // In production, map specific CUDA errors to a custom error category
  return std::make_error_code(std::errc::resource_unavailable_try_again);
}

class CUDATask : public Task {
private:
  cudaStream_t stream_;
  cudaEvent_t event_;
  const Device *device_;

public:
  template <typename Func, typename... Args>
  explicit CUDATask(Func &&func, const Device *device, Args &&...args) : device_(device) {
    cudaStreamCreate(&stream_);

    cudaEventCreate(&event_);

    auto launch_func = [f = std::forward<Func>(func),
                        args_tuple = std::make_tuple(std::forward<Args>(args)...),
                        stream = stream_]() mutable {
      std::apply([stream](auto &&f_inner,
                          auto &&...a) { f_inner(std::forward<decltype(a)>(a)..., stream); },
                 std::tuple_cat(std::forward_as_tuple(f), std::move(args_tuple)));
    };

    launch_func();
    cudaEventRecord(event_, stream_);
  }

  ~CUDATask() {
    cudaStreamDestroy(stream_);
    cudaEventDestroy(event_);
  }

  ErrorStatus synchronize() override {
    cudaError_t err = cudaEventSynchronize(event_);

    return cuda_error_to_status(err);
  }
};

#endif // USE_CUDA

inline void wait_all(std::vector<std::unique_ptr<Task>> &contexts) {
  for (auto &ctx : contexts) {
    ErrorStatus status = ctx->synchronize();
    if (status) {
      throw std::runtime_error("Synchronization failed: " + status.message());
    }
  }
}
} // namespace tnn