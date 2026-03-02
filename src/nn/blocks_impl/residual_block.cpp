/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "nn/blocks_impl/residual_block.hpp"

#include <algorithm>
#include <numeric>

#include "nn/activations.hpp"
#include "nn/layers.hpp"
#include "ops/ops.hpp"
#include "tensor/tensor.hpp"

namespace tnn {

ResidualBlock::ResidualBlock(std::vector<std::unique_ptr<Layer>> main_path,
                             std::vector<std::unique_ptr<Layer>> shortcut_path,
                             const std::string &final_activation, const std::string &name)
    : main_path_(std::move(main_path)),
      shortcut_path_(std::move(shortcut_path)),
      activation_type_(final_activation) {
  if (main_path_.empty()) {
    throw std::runtime_error("Main path of ResidualBlock cannot be empty.");
  }
  this->name_ = name;

  if (final_activation != "none" && final_activation != "linear") {
    auto factory = ActivationFactory();
    factory.register_defaults();
    final_activation_ = factory.create(final_activation);
  }
}

static size_t compute_path_max_size(const std::vector<std::unique_ptr<Layer>> &path,
                                    const std::vector<size_t> &input_shape, DType_t dtype) {
  size_t max_size = 0;
  std::vector<size_t> current_shape = input_shape;
  for (const auto &layer : path) {
    current_shape = layer->output_shape({current_shape})[0];
    size_t layer_size =
        std::accumulate(current_shape.begin(), current_shape.end(), 1, std::multiplies<size_t>());
    max_size = std::max(max_size, layer_size);
  }
  return max_size;
}

void ResidualBlock::forward(const Vec<ConstTensor> &inputs, const Vec<Tensor> &outputs,
                            size_t mb_id) {
  const ConstTensor &input = inputs[0];
  const Tensor &output = outputs[0];

  input_shape_cache_[mb_id] = input->shape();

  ConstTensor main_output = input;  // main output = f exist ? input : f(input)
  for (auto &layer : main_path_) {
    Tensor temp_output =
        this->get_buffer(layer->output_shape({main_output->shape()})[0], main_output->data_type());
    layer->forward({main_output}, {temp_output}, mb_id);
    main_output = temp_output;
  }

  ConstTensor shortcut_output = input;  // shortcut output = g exist ? input : g(input)
  for (auto &layer : shortcut_path_) {
    Tensor temp_output = this->get_buffer(layer->output_shape({shortcut_output->shape()})[0],
                                          shortcut_output->data_type());
    layer->forward({shortcut_output}, {temp_output}, mb_id);
    shortcut_output = temp_output;
  }

  if (final_activation_) {
    Tensor &pre_act = this->get_mutable_tensor(mb_id, "pre_activation");
    if (!pre_act)
      pre_act = this->get_buffer(main_output->shape(), main_output->data_type());
    else
      pre_act->ensure(main_output->shape());
    DISPATCH_IO_DTYPE(ops::add, main_output->data_ptr(), shortcut_output->data_ptr(),
                      pre_act->data_ptr(), pre_act->size());

    output->ensure(main_output->shape());
    final_activation_->apply(pre_act, output);
  } else {
    output->ensure(main_output->shape());
    DISPATCH_IO_DTYPE(ops::add, main_output->data_ptr(), shortcut_output->data_ptr(),
                      output->data_ptr(), output->size());
  }
}

void ResidualBlock::backward(const Vec<ConstTensor> &grad_outputs, const Vec<Tensor> &grad_inputs,
                             size_t mb_id) {
  const ConstTensor &grad_output = grad_outputs[0];
  const Tensor &grad_input = grad_inputs[0];
  const Tensor &pre_act = this->get_mutable_tensor(mb_id, "pre_activation");
  if (final_activation_ && !pre_act) {
    throw std::runtime_error("No cached pre-activation output found for micro-batch ID: " +
                             std::to_string(mb_id));
  }

  ConstTensor grad_to_propagate = grad_output;

  if (final_activation_) {
    Tensor dpre_act = this->get_buffer(pre_act->shape(), pre_act->data_type());
    final_activation_->compute_gradient(pre_act, grad_output, dpre_act);
    grad_to_propagate = dpre_act;
  }

  auto it_input_shape = input_shape_cache_.find(mb_id);
  if (it_input_shape == input_shape_cache_.end()) {
    throw std::runtime_error("No cached input shape found for micro-batch ID: " +
                             std::to_string(mb_id));
  }

  size_t main_path_max_size =
      compute_path_max_size(main_path_, it_input_shape->second, grad_output->data_type());
  size_t shortcut_path_max_size =
      compute_path_max_size(shortcut_path_, it_input_shape->second, grad_output->data_type());

  // little trick to avoid const correctness issue
  ConstTensor main_grad = grad_to_propagate;
  for (int i = static_cast<int>(main_path_.size()) - 1; i >= 0; --i) {
    Tensor temp_grad = this->get_buffer({main_path_max_size}, grad_output->data_type());
    main_path_[i]->backward({main_grad}, {temp_grad}, mb_id);
    main_grad = temp_grad;
  }

  ConstTensor shortcut_grad = grad_to_propagate;  // same here
  if (!shortcut_path_.empty()) {
    for (int i = static_cast<int>(shortcut_path_.size()) - 1; i >= 0; --i) {
      Tensor temp_grad = this->get_buffer({shortcut_path_max_size}, grad_output->data_type());
      shortcut_path_[i]->backward({shortcut_grad}, {temp_grad}, mb_id);
      shortcut_grad = temp_grad;
    }
  }

  grad_input->ensure(main_grad->shape());
  DISPATCH_IO_DTYPE(ops::add, main_grad->data_ptr(), shortcut_grad->data_ptr(),
                    grad_input->data_ptr(), grad_input->size());
}

Vec<Vec<size_t>> ResidualBlock::output_shape(const Vec<Vec<size_t>> &input_shapes) const {
  std::vector<size_t> shape = input_shapes[0];
  for (const auto &layer : main_path_) {
    shape = layer->output_shape({shape})[0];
  }
  return {shape};
}

LayerConfig ResidualBlock::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.type = this->type();
  config.set("activation", activation_type_);

  nlohmann::json main_array = nlohmann::json::array();
  for (const auto &layer : main_path_) {
    LayerConfig sub_cfg = layer->get_config();
    nlohmann::json sub_json = sub_cfg.to_json();
    main_array.push_back(sub_json);
  }

  nlohmann::json shortcut_array = nlohmann::json::array();
  for (const auto &layer : shortcut_path_) {
    LayerConfig sub_cfg = layer->get_config();
    nlohmann::json sub_json = sub_cfg.to_json();
    shortcut_array.push_back(sub_json);
  }

  config.set("main_path", main_array);
  config.set("shortcut_path", shortcut_array);

  return config;
}

std::unique_ptr<ResidualBlock> ResidualBlock::create_from_config(const LayerConfig &config) {
  std::vector<std::unique_ptr<Layer>> main_path;
  std::vector<std::unique_ptr<Layer>> shortcut_path;
  nlohmann::json main_json = config.get<nlohmann::json>("main_path", nlohmann::json::array());
  LayerFactory::register_defaults();
  for (const auto &layer_json : main_json) {
    LayerConfig layer_config = LayerConfig::from_json(layer_json);
    auto layer = LayerFactory::create(layer_config);
    main_path.push_back(std::move(layer));
  }
  nlohmann::json shortcut_json =
      config.get<nlohmann::json>("shortcut_path", nlohmann::json::array());
  for (const auto &layer_json : shortcut_json) {
    LayerConfig layer_config = LayerConfig::from_json(layer_json);
    auto layer = LayerFactory::create(layer_config);
    shortcut_path.push_back(std::move(layer));
  }

  std::string activation = config.get<std::string>("activation", "relu");
  return std::make_unique<ResidualBlock>(std::move(main_path), std::move(shortcut_path), activation,
                                         config.name);
}

size_t ResidualBlock::fwd_workspace(const Vec<Vec<size_t>> &input_shapes) const {
  if (main_path_.empty()) return 0;
  const auto &input_shape = input_shapes[0];
  size_t total_ws = 0;
  size_t dtype_size = get_dtype_size(io_dtype_);

  // Main path workspace
  Vec<size_t> main_shape = input_shape;
  for (const auto &layer : main_path_) {
    total_ws += layer->fwd_workspace({{main_shape}});
    total_ws += std::accumulate(main_shape.begin(), main_shape.end(), dtype_size,
                                std::multiplies<size_t>());
    main_shape = layer->output_shape({main_shape})[0];
  }

  // Shortcut path workspace
  Vec<size_t> shortcut_shape = input_shape;
  for (const auto &layer : shortcut_path_) {
    total_ws += layer->fwd_workspace({{shortcut_shape}});
    total_ws += std::accumulate(shortcut_shape.begin(), shortcut_shape.end(), dtype_size,
                                std::multiplies<size_t>());
    shortcut_shape = layer->output_shape({shortcut_shape})[0];
  }

  // Pre-activation buffer if final activation exists
  if (final_activation_) {
    total_ws += std::accumulate(main_shape.begin(), main_shape.end(), dtype_size,
                                std::multiplies<size_t>());
  }

  return total_ws;
}

size_t ResidualBlock::inf_workspace(const Vec<Vec<size_t>> &input_shapes) const {
  if (main_path_.empty()) return 0;
  const auto &input_shape = input_shapes[0];
  size_t dtype_size = get_dtype_size(io_dtype_);

  // Helper lambda to compute buffer cycling workspace for a path (SISO sequence)
  auto compute_path_workspace = [dtype_size](const std::vector<std::unique_ptr<Layer>> &path,
                                             const Vec<size_t> &input_shape) -> size_t {
    if (path.empty()) return 0;
    Vec<size_t> out_bytes;
    Vec<size_t> sub_ws;
    Vec<size_t> cur = input_shape;
    for (const auto &layer : path) {
      Vec<size_t> out = layer->output_shape({cur})[0];
      size_t bytes =
          std::accumulate(out.begin(), out.end(), dtype_size, std::multiplies<size_t>());
      out_bytes.push_back(bytes);
      sub_ws.push_back(layer->inf_workspace({{cur}}));
      cur = out;
    }
    size_t m_b = 0;
    for (size_t i = 0; i < out_bytes.size() - 1; ++i) {
      m_b = std::max(m_b, out_bytes[i] + out_bytes[i + 1] + sub_ws[i]);
    }
    // Don't forget the last layer's workspace
    if (!out_bytes.empty() && !sub_ws.empty()) {
      m_b = std::max(m_b, out_bytes.back() + sub_ws.back());
    }
    return m_b;
  };

  // Compute O_i (terminal output size) for each path
  Vec<size_t> main_output_shape = input_shape;
  for (const auto &layer : main_path_) {
    main_output_shape = layer->output_shape({main_output_shape})[0];
  }
  size_t main_output_bytes =
      std::accumulate(main_output_shape.begin(), main_output_shape.end(), dtype_size,
                      std::multiplies<size_t>());

  Vec<size_t> shortcut_output_shape = input_shape;
  for (const auto &layer : shortcut_path_) {
    shortcut_output_shape = layer->output_shape({shortcut_output_shape})[0];
  }
  size_t shortcut_output_bytes =
      std::accumulate(shortcut_output_shape.begin(), shortcut_output_shape.end(), dtype_size,
                      std::multiplies<size_t>());

  // Compute M_b,i (buffer cycling workspace) for each path
  size_t main_workspace = compute_path_workspace(main_path_, input_shape);
  size_t shortcut_workspace = compute_path_workspace(shortcut_path_, input_shape);

  // Apply Algorithm 2 from paper (MISO joining)
  // We have 2 sequences with (a_i, b_i) = (M_b,i, O_i)
  // Sort by (a_i - b_i) descending to find optimal order
  struct PathInfo {
    size_t workspace;      // a_i
    size_t output_bytes;   // b_i
    size_t priority;       // a_i - b_i
  };

  PathInfo main_info{main_workspace, main_output_bytes, main_workspace - main_output_bytes};
  PathInfo shortcut_info{shortcut_workspace, shortcut_output_bytes,
                         shortcut_workspace - shortcut_output_bytes};

  // Execute in optimal order and compute peak memory
  size_t k = 0;
  if (main_info.priority >= shortcut_info.priority) {
    // Execute main first, then shortcut
    k = std::max(main_info.workspace, shortcut_info.workspace + main_info.output_bytes);
  } else {
    // Execute shortcut first, then main
    k = std::max(shortcut_info.workspace, main_info.workspace + shortcut_info.output_bytes);
  }

  // Add space for the join operation (add + optional activation)
  size_t join_output_bytes =
      std::accumulate(main_output_shape.begin(), main_output_shape.end(), dtype_size,
                      std::multiplies<size_t>());
  k = std::max(k, main_output_bytes + shortcut_output_bytes + join_output_bytes);

  return k;
}

size_t ResidualBlock::bwd_workspace(const Vec<Vec<size_t>> &input_shapes) const {
  if (main_path_.empty()) return 0;
  const auto &input_shape = input_shapes[0];
  size_t dtype_size = get_dtype_size(io_dtype_);

  // Helper lambda to compute buffer cycling workspace for a path (SISO sequence)
  auto compute_path_workspace = [dtype_size](const std::vector<std::unique_ptr<Layer>> &path,
                                             const Vec<size_t> &input_shape) -> size_t {
    if (path.empty()) return 0;
    Vec<size_t> out_bytes;
    Vec<size_t> sub_ws;
    Vec<size_t> cur = input_shape;
    for (const auto &layer : path) {
      Vec<size_t> out = layer->output_shape({cur})[0];
      size_t bytes =
          std::accumulate(out.begin(), out.end(), dtype_size, std::multiplies<size_t>());
      out_bytes.push_back(bytes);
      sub_ws.push_back(layer->bwd_workspace({{cur}}));
      cur = out;
    }
    size_t m_b = 0;
    for (size_t i = 0; i < out_bytes.size() - 1; ++i) {
      m_b = std::max(m_b, out_bytes[i] + out_bytes[i + 1] + sub_ws[i]);
    }
    // Don't forget the last layer's workspace
    if (!out_bytes.empty() && !sub_ws.empty()) {
      m_b = std::max(m_b, out_bytes.back() + sub_ws.back());
    }
    return m_b;
  };

  // Compute O_i (terminal output size) for each path
  Vec<size_t> main_output_shape = input_shape;
  for (const auto &layer : main_path_) {
    main_output_shape = layer->output_shape({main_output_shape})[0];
  }
  size_t main_output_bytes =
      std::accumulate(main_output_shape.begin(), main_output_shape.end(), dtype_size,
                      std::multiplies<size_t>());

  Vec<size_t> shortcut_output_shape = input_shape;
  for (const auto &layer : shortcut_path_) {
    shortcut_output_shape = layer->output_shape({shortcut_output_shape})[0];
  }
  size_t shortcut_output_bytes =
      std::accumulate(shortcut_output_shape.begin(), shortcut_output_shape.end(), dtype_size,
                      std::multiplies<size_t>());

  // Compute M_b,i (buffer cycling workspace) for each path
  size_t main_workspace = compute_path_workspace(main_path_, input_shape);
  size_t shortcut_workspace = compute_path_workspace(shortcut_path_, input_shape);

  // Apply Algorithm 2 from paper (MISO joining)
  // We have 2 sequences with (a_i, b_i) = (M_b,i, O_i)
  // Sort by (a_i - b_i) descending to find optimal order
  struct PathInfo {
    size_t workspace;      // a_i
    size_t output_bytes;   // b_i
    size_t priority;       // a_i - b_i
  };

  PathInfo main_info{main_workspace, main_output_bytes, main_workspace - main_output_bytes};
  PathInfo shortcut_info{shortcut_workspace, shortcut_output_bytes,
                         shortcut_workspace - shortcut_output_bytes};

  // Execute in optimal order and compute peak memory
  size_t k = 0;
  if (main_info.priority >= shortcut_info.priority) {
    // Execute main first, then shortcut
    k = std::max(main_info.workspace, shortcut_info.workspace + main_info.output_bytes);
  } else {
    // Execute shortcut first, then main
    k = std::max(shortcut_info.workspace, main_info.workspace + shortcut_info.output_bytes);
  }

  // Add space for the backward through join operation
  size_t grad_input_bytes =
      std::accumulate(input_shape.begin(), input_shape.end(), dtype_size, std::multiplies<size_t>());
  k = std::max(k, main_output_bytes + shortcut_output_bytes + grad_input_bytes);

  return k;
}

}  // namespace tnn
