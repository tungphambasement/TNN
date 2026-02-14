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

void ResidualBlock::forward_impl(const ConstTensor &input, const Tensor &output, size_t mb_id) {
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

void ResidualBlock::backward_impl(const ConstTensor &grad_output, const Tensor &grad_input,
                                  size_t mb_id) {
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

std::vector<size_t> ResidualBlock::compute_output_shape(
    const std::vector<size_t> &input_shape) const {
  std::vector<size_t> shape = input_shape;
  for (const auto &layer : main_path_) {
    shape = layer->output_shape({shape})[0];
  }
  return shape;
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

}  // namespace tnn
