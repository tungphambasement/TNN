/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "nn/layer.hpp"

#include "logging/logger.hpp"

namespace tnn {

void Layer::forward(const std::vector<ConstTensor> &inputs, const std::vector<Tensor> &outputs,
                    size_t mb_id) {
  if (!initialized_) {
    std::cerr << "Warning: Layer " << name_ << " is not initialized. Call init() before forward."
              << std::endl;
    return;
  }
  Clock::time_point start_time = Clock::now();
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
  Clock::time_point end_time = Clock::now();
  profiler_.add_event(Event{EventType::COMPUTE, start_time, end_time, "forward"});
}

void Layer::backward(const std::vector<ConstTensor> &gradients,
                     const std::vector<Tensor> &grad_inputs, size_t mb_id) {
  if (!initialized_) {
    std::cerr << "Warning: Layer " << name_ << " is not initialized. Call init() before backward."
              << std::endl;
    return;
  }
  auto start_time = Clock::now();
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
  Clock::time_point end_time = Clock::now();
  profiler_.add_event(Event{EventType::COMPUTE, start_time, end_time, "backward"});
}

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

Layer &Layer::enable_profiling(bool enable) {
  enable_profiling_ = enable;
  return *this;
}

bool Layer::is_profiling_enabled() const { return enable_profiling_; }

void Layer::print_profiling() const {
  const auto &events = profiler_.get_events();
  std::string header = "Profiling info for Layer: " + name_;
  std::string format_str = "{:=^80}\n";
  std::string output = fmt::format(fmt::runtime(format_str), header);
  output += fmt::format(fmt::runtime("{:<30}{:<30}{:<20}\n"), "Event", "Duration (ms)", "");
  output += fmt::format(fmt::runtime("{:-^80}\n"), "");
  for (const auto &event : events) {
    float duration_ms =
        Time::duration_cast<Time::microseconds>(event.end_time - event.start_time).count() /
        1000.0f;
    output += fmt::format(fmt::runtime("{:<30}{:<30.3f}\n"), event.name, duration_ms);
  }
  output += fmt::format(fmt::runtime("{:=^80}"), "");
  GlobalLogger::info(output);
}

size_t Layer::cached_memory_bytes() const {
  size_t total = 0;
  for (auto &[key, tensor] : cached_tensors_) {
    if (tensor) {
      size_t dtype_size = get_dtype_size(tensor->data_type());
      total += tensor->capacity() * dtype_size;
    }
  }
  return total;
}

void Layer::reset_profiling() { profiler_.reset(); }

std::string Layer::name() const { return name_; }

void Layer::save_state(std::ofstream &file) {
  auto config = get_config();
  nlohmann::json j = config.to_json();
  std::string j_str = j.dump();
  size_t j_size = j_str.size();
  file.write(reinterpret_cast<const char *>(&j_size), sizeof(size_t));
  file.write(j_str.c_str(), j_size);
  auto params = parameters();
  for (const auto &param : params) {
    param->save(file);
  }
}

void Layer::register_param(std::vector<size_t> shape) {
  if (!context_) {
    throw std::runtime_error("Context is not set");
  }
  context_->register_param(shape, param_dtype_);
}

Tensor Layer::make_param_tensor(std::vector<size_t> shape) {
  if (!context_) {
    throw std::runtime_error("Context is not set");
  }
  Tensor param = context_->get_param(shape, param_dtype_);
  params_.push_back(param);
  return param;
}

Tensor Layer::make_grad_tensor(std::vector<size_t> shape) {
  if (!context_) {
    throw std::runtime_error("Context is not set");
  }
  Tensor grad = context_->get_grad(shape, param_dtype_);
  grads_.push_back(grad);
  return grad;
}

Tensor Layer::make_io_tensor(std::vector<size_t> shape) {
  if (!context_) {
    throw std::runtime_error("Context is not set");
  }
  return context_->get_workspace(shape, io_dtype_);
}

Tensor Layer::make_compute_tensor(std::vector<size_t> shape) {
  if (!context_) {
    throw std::runtime_error("Context is not set");
  }
  return context_->get_workspace(shape, compute_dtype_);
}

ConstTensor &Layer::get_cached_tensor(size_t mb_id, const std::string &key) {
  return cached_tensors_[{mb_id, key}];
}

Tensor &Layer::get_mutable_tensor(size_t mb_id, const std::string &key) {
  return mutable_tensors_[{mb_id, key}];
}

Tensor Layer::get_buffer(const std::vector<size_t> &shape, DType_t dtype) {
  if (!context_) {
    throw std::runtime_error("Context is not set");
  }
  return context_->get_workspace(shape, dtype);
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
