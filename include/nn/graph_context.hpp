#pragma once

#include <vector>

#include "device/device.hpp"
#include "device/iallocator.hpp"
#include "ops/ops.hpp"
#include "tensor/tensor.hpp"
#include "type/type.hpp"

namespace tnn {
class GraphContext {
public:
  GraphContext(IAllocator &allocator)
      : allocator_(allocator) {}

  ~GraphContext() = default;

  GraphContext(const GraphContext &) = delete;
  GraphContext &operator=(const GraphContext &) = delete;

  GraphContext(GraphContext &&) = default;
  GraphContext &operator=(GraphContext &&) = delete;

  void register_param(std::vector<size_t> shape, DType_t dtype) {
    size_t bytes_size = get_bytes_size(shape, dtype);
    // round up to 256 bytes for better memory access pattern, can be tuned later
    bytes_size = (bytes_size + 255) & ~255;

    // assuming param and grad have the same shape and dtype for simplicity
    param_bytes += bytes_size;
    grad_bytes += bytes_size;
  }

  void init() {
    if (initialized_) {
      throw std::runtime_error(
          "Cannot initalize GraphContext more than once. Possible dangling reference to old "
          "context can happen.");
    }
    initialized_ = true;
    param_slab_ = allocator_.allocate(param_bytes);
    grad_slab_ = allocator_.allocate(grad_bytes);
  }

  Tensor get_param(std::vector<size_t> shape, DType_t dtype) {
    if (!initialized_) {
      throw std::runtime_error("GraphContext must be initialized before getting parameters.");
    }
    size_t bytes_size = get_bytes_size(shape, dtype);
    // round up to 256 bytes for better memory access pattern, can be tuned later
    bytes_size = (bytes_size + 255) & ~255;
    dptr param_data = param_slab_.span(param_offset_, bytes_size);
    param_offset_ += bytes_size;
    Tensor param = make_tensor(allocator_, dtype, std::move(param_data), shape);
    params_.push_back(param);
    return param;
  }

  Tensor get_grad(std::vector<size_t> shape, DType_t dtype) {
    if (!initialized_) {
      throw std::runtime_error("GraphContext must be initialized before getting gradients.");
    }
    size_t bytes_size = get_bytes_size(shape, dtype);
    // round up to 256 bytes for better memory access pattern, can be tuned later
    bytes_size = (bytes_size + 255) & ~255;
    dptr grad_data = grad_slab_.span(grad_offset_, bytes_size);
    grad_offset_ += bytes_size;
    Tensor grad = make_tensor(allocator_, dtype, std::move(grad_data), shape);
    grads_.push_back(grad);
    return grad;
  }

  Tensor get_workspace(std::vector<size_t> shape, DType_t dtype) {
    // for now, allocate from pool
    return make_tensor(allocator_, dtype, shape);
  }

  std::vector<Tensor> parameters() { return params_; }
  std::vector<Tensor> gradients() { return grads_; }

  void clear_gradients() {
    if (!initialized_) {
      throw std::runtime_error("GraphContext must be initialized before clearing gradients.");
    }
    ops::set_scalar<uchar>(grad_slab_, 0, grad_bytes);
  }

  IAllocator &allocator() { return allocator_; }

  const Device &device() const { return allocator_.device(); }

private:
  bool initialized_ = false;
  IAllocator &allocator_;
  size_t param_bytes = 0;
  size_t grad_bytes = 0;
  std::vector<Tensor> params_, grads_;
  dptr param_slab_, grad_slab_;
  size_t param_offset_ = 0, grad_offset_ = 0;

  static size_t get_bytes_size(const std::vector<size_t> &shape, DType_t dtype) {
    return std::accumulate(shape.begin(), shape.end(), get_dtype_size(dtype),
                           std::multiplies<size_t>());
  }
};
}  // namespace tnn