/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <fmt/core.h>

#include <cstddef>
#include <cstring>

#include "common/config.hpp"
#include "nn/node.hpp"
#include "profiling/profiler.hpp"
#include "tensor/tensor.hpp"
#include "type/type.hpp"

namespace tnn {
using LayerConfig = TConfig;

struct ParamDescriptor {
  std::vector<size_t> shape;
  Tensor *data_ptr;  // pointer to the member variable in concrete layer
  Tensor *grad_ptr;  // pointer to the member variable in concrete layer
};

// Single input/output layer interface. For multi-input/output, use a Block to wrap multiple Layers.
class Layer : public INode {
public:
  Layer() = default;

  // Note: have to call init again after changing param dtype
  Layer &set_flow_handle(flowHandle_t handle);
  flowHandle_t get_flow_handle() const;
  Layer &set_seed(unsigned long long seed);
  Layer &set_io_dtype(DType_t dtype);
  DType_t get_io_dtype() const;
  Layer &set_param_dtype(DType_t dtype);
  DType_t get_param_dtype() const;
  Layer &set_compute_dtype(DType_t dtype);
  DType_t get_compute_dtype() const;
  Layer &set_training(bool training);
  bool is_training() const;
  Layer &enable_profiling(bool enable);
  bool is_profiling_enabled() const;

  virtual uint64_t forward_flops(const std::vector<size_t> &input_shape) const = 0;
  virtual uint64_t backward_flops(const std::vector<size_t> &input_shape) const = 0;
  virtual std::vector<size_t> compute_output_shape(
      const std::vector<size_t> &input_shape) const = 0;
  void print_profiling() const;
  void reset_profiling();
  virtual size_t cached_memory_bytes() const;
  std::string name() const;

  void forward(const std::vector<ConstTensor> &inputs, const std::vector<Tensor> &outputs,
               size_t mb_id = 0);
  void backward(const std::vector<ConstTensor> &gradients, const std::vector<Tensor> &grad_inputs,
                size_t mb_id = 0);

  std::vector<Tensor> parameters() const { return params_; }
  std::vector<Tensor> gradients() const { return grads_; }

  void save_state(std::ofstream &file) override;

protected:
  virtual std::vector<ParamDescriptor> param_descriptors() const { return {}; }
  virtual void on_set_seed(unsigned long long seed) {}
  virtual void on_set_training(bool training) {}
  virtual void on_set_io_dtype(DType_t dtype) {}
  virtual void on_set_param_dtype(DType_t dtype) {}
  virtual void on_set_compute_dtype(DType_t dtype) {}
  virtual void forward_impl(const ConstTensor &input, const Tensor &output, size_t mb_id = 0) = 0;
  virtual void backward_impl(const ConstTensor &gradient, const Tensor &grad_input,
                             size_t mb_id = 0) = 0;

protected:
  bool is_training_ = true;
  bool enable_profiling_ = false;
  bool use_seed_ = false;
  unsigned long long srand_seed_ = 0;
  std::map<std::pair<size_t, std::string>, Tensor> mutable_tensors_;
  // for immutable cache (e.g., inputs)
  std::map<std::pair<size_t, std::string>, ConstTensor> cached_tensors_;
  Profiler profiler_;
  flowHandle_t flow_handle_;
  std::string name_;
  DType_t io_dtype_ = DType_t::FP32;       // data type for input/output tensors
  DType_t param_dtype_ = DType_t::FP32;    // data type for parameters/gradients
  DType_t compute_dtype_ = DType_t::FP32;  // data type for internal computations
  std::vector<Tensor> params_, grads_;

  // helpers
  void register_param(std::vector<size_t> shape);
  Tensor make_param_tensor(std::vector<size_t> shape);
  Tensor make_grad_tensor(std::vector<size_t> shape);
  Tensor make_io_tensor(std::vector<size_t> shape);
  Tensor make_compute_tensor(std::vector<size_t> shape);
  ConstTensor &get_cached_tensor(size_t mb_id, const std::string &key);
  Tensor &get_mutable_tensor(size_t mb_id, const std::string &key);
  Tensor get_buffer(const std::vector<size_t> &shape, DType_t dtype = DType_t::FP32);

  void clear_cache(size_t mb_id);
};

#define DISPATCH_IO_DTYPE(method_name, ...)                                \
  do {                                                                     \
    DISPATCH_DTYPE(this->io_dtype_, IO_T, method_name<IO_T>(__VA_ARGS__)); \
  } while (0)

#define DISPATCH_ON_3_DTYPES_TO_METHOD(method_name, ...)                                   \
  do {                                                                                     \
    DISPATCH_DTYPE(                                                                        \
        this->io_dtype_, IO_T,                                                             \
        DISPATCH_DTYPE(this->param_dtype_, PARAM_T,                                        \
                       DISPATCH_DTYPE(this->compute_dtype_, COMP_T,                        \
                                      method_name<IO_T, PARAM_T, COMP_T>(__VA_ARGS__);))); \
  } while (0)
}  // namespace tnn