/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "nn/layer.hpp"

#include <numeric>

#include "type/type.hpp"

namespace tnn {

void Layer::init() {
  if (initialized_) {
    throw std::runtime_error("Cannot initalize Layer more than once. ");
  }
  init_impl();
  initialized_ = true;
}

void Layer::forward(const std::vector<ConstTensor> &inputs, const std::vector<Tensor> &outputs,
                    size_t mb_id) {
  if (!initialized_) {
    std::cerr << "Warning: Layer " << name_ << " is not initialized. Call init() before forward."
              << std::endl;
    return;
  }
  if (inputs.empty() || outputs.empty()) {
    throw std::runtime_error("Layer " + name_ + " received empty IO tensors.");
  }
  if (inputs[0]->data_type() != io_dtype_) {
    throw std::runtime_error("Layer " + name_ +
                             " input tensor dtype does not match layer io_dtype.");
  }
  if (outputs[0]->data_type() != io_dtype_) {
    throw std::runtime_error("Layer " + name_ +
                             " output tensor dtype does not match layer io_dtype.");
  }
  ConstTensor current = inputs[0];
  Tensor device_input;
  if (inputs[0]->device() != this->device()) {
    device_input = this->get_buffer(inputs[0]->shape(), inputs[0]->data_type());
    inputs[0]->copy_to(device_input);
    current = device_input;
  }
  if (outputs[0]->device() != this->device()) {
    throw std::runtime_error("Layer " + name_ +
                             " output tensor device does not match layer device.");
  }
  forward_impl(current, outputs[0], mb_id);
#ifndef NDEBUG
  this->device().getFlow(this->flow_handle_)->synchronize();
#endif
}

void Layer::backward(const std::vector<ConstTensor> &gradients,
                     const std::vector<Tensor> &grad_inputs, size_t mb_id) {
  if (!initialized_) {
    std::cerr << "Warning: Layer " << name_ << " is not initialized. Call init() before backward."
              << std::endl;
    return;
  }
  if (gradients.empty() || grad_inputs.empty()) {
    throw std::runtime_error("Layer " + name_ +
                             " received empty gradients or grad_inputs tensors.");
  }
  if (gradients[0]->data_type() != io_dtype_) {
    throw std::runtime_error("Layer " + name_ +
                             " gradient tensor dtype does not match layer io_dtype.");
  }
  if (grad_inputs[0]->data_type() != io_dtype_) {
    throw std::runtime_error("Layer " + name_ +
                             " grad_input tensor dtype does not match layer io_dtype.");
  }
  ConstTensor current_gradient = gradients[0];
  Tensor device_gradient;
  if (gradients[0]->device() != this->device()) {
    device_gradient = this->get_buffer(gradients[0]->shape(), gradients[0]->data_type());
    gradients[0]->copy_to(device_gradient);
    current_gradient = device_gradient;
  }
  if (grad_inputs[0]->device() != this->device()) {
    throw std::runtime_error("Layer " + name_ +
                             " grad_input tensor device does not match layer device.");
  }
  backward_impl(current_gradient, grad_inputs[0], mb_id);
#ifndef NDEBUG
  this->device().getFlow(this->flow_handle_)->synchronize();
#endif
  clear_cache(mb_id);
}

Layer &Layer::set_allocator(IAllocator &allocator) {
  allocator_ = &allocator;
  on_set_allocator(allocator);
  return *this;
}

IAllocator *Layer::get_allocator() const { return allocator_; }

Layer &Layer::set_flow_handle(flowHandle_t handle) {
  flow_handle_ = handle;
  on_set_flow_handle(handle);
  return *this;
}

flowHandle_t Layer::get_flow_handle() const { return flow_handle_; }

Layer &Layer::set_io_dtype(DType_t dtype) {
  io_dtype_ = dtype;
  on_set_io_dtype(dtype);
  return *this;
}

DType_t Layer::get_io_dtype() const { return io_dtype_; }

Layer &Layer::set_param_dtype(DType_t dtype) {
  param_dtype_ = dtype;
  on_set_param_dtype(dtype);
  return *this;
}

DType_t Layer::get_param_dtype() const { return param_dtype_; }

Layer &Layer::set_compute_dtype(DType_t dtype) {
  compute_dtype_ = dtype;
  on_set_compute_dtype(dtype);
  return *this;
}

DType_t Layer::get_compute_dtype() const { return compute_dtype_; }

Layer &Layer::set_seed(unsigned long long seed) {
  use_seed_ = true;
  srand_seed_ = seed;
  on_set_seed(seed);
  return *this;
}

Layer &Layer::set_training(bool training) {
  is_training_ = training;
  on_set_training(training);
  return *this;
}

bool Layer::is_training() const { return is_training_; }

void Layer::save_state(std::ofstream &file) {
  auto config = get_config();
  nlohmann::json j = config.to_json();
  std::string j_str = j.dump();
  size_t j_size = j_str.size();
  file.write(reinterpret_cast<const char *>(&j_size), sizeof(size_t));
  file.write(j_str.c_str(), j_size);
  auto descs = param_descriptors();
  for (const auto &desc : descs) {
    Tensor param = *desc.data_ptr;
    param->save(file);
  }
}

std::vector<Tensor> Layer::parameters() {
  std::vector<Tensor> params;
  auto descs = this->param_descriptors();
  for (const auto &desc : descs) {
    params.push_back(*desc.data_ptr);
  }
  return params;
}

std::vector<Tensor> Layer::gradients() {
  std::vector<Tensor> grads;
  auto descs = this->param_descriptors();
  for (const auto &desc : descs) {
    grads.push_back(*desc.grad_ptr);
  }
  return grads;
}

Vec<Vec<size_t>> Layer::output_shape(const Vec<Vec<size_t>> &input_shape) const {
  if (input_shape.size() != 1) {
    throw std::runtime_error("Currently only single input supported in output_shape.");
  }
  return {compute_output_shape(input_shape[0])};
}

Tensor Layer::make_io_tensor(std::vector<size_t> shape) { return get_buffer(shape, io_dtype_); }

Tensor Layer::make_compute_tensor(std::vector<size_t> shape) {
  return get_buffer(shape, compute_dtype_);
}

ConstTensor &Layer::get_cached_tensor(size_t mb_id, const std::string &key) {
  return cached_tensors_[{mb_id, key}];
}

Tensor &Layer::get_mutable_tensor(size_t mb_id, const std::string &key) {
  return mutable_tensors_[{mb_id, key}];
}

Tensor Layer::get_buffer(const std::vector<size_t> &shape, DType_t dtype) {
  if (!allocator_) {
    throw std::runtime_error("Allocator is not set");
  }
  auto byte_size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>()) *
                   get_dtype_size(dtype);
  dptr buffer = allocator_->allocate(byte_size);
  return make_tensor(*allocator_, dtype, std::move(buffer), shape);
}

void Layer::clear_cache(size_t mb_id) {
  for (auto it = cached_tensors_.begin(); it != cached_tensors_.end();) {
    if (it->first.first == mb_id) {
      it = cached_tensors_.erase(it);
    } else {
      ++it;
    }
  }
}

}  // namespace tnn
