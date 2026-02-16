/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "nn/blocks_impl/sequential.hpp"

#include <fmt/core.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>

#include "nlohmann/json_fwd.hpp"
#include "nn/block.hpp"
#include "nn/layers.hpp"
#include "nn/siso_layer.hpp"

namespace tnn {

void Sequential::compute_max_size(const std::vector<size_t> &input_shape, DType_t dtype) {
  std::vector<size_t> current_shape = input_shape;
  size_t current_max =
      std::accumulate(current_shape.begin(), current_shape.end(), 1, std::multiplies<size_t>());
  for (const auto &layer : layers_) {
    current_shape = layer->output_shape({current_shape})[0];
    size_t activation_size =
        std::accumulate(current_shape.begin(), current_shape.end(), 1, std::multiplies<size_t>());
    current_max = std::max(current_max, activation_size);
  }
  max_size_ = current_max;
}

void Sequential::forward_impl(const ConstTensor &input, const Tensor &output, size_t mb_id) {
  if (layers_.empty()) {
    throw std::runtime_error("Cannot forward through empty sequential model");
  }
  compute_max_size(input->shape(), input->data_type());
  ConstTensor current_input = input;
  Tensor current_output = nullptr;
  for (size_t i = 0; i < layers_.size(); ++i) {
    try {
      if (i < layers_.size() - 1) {
        current_output = this->get_buffer(layers_[i]->output_shape({current_input->shape()})[0],
                                          input->data_type());
      } else {
        current_output = output;
      }
      layers_[i]->forward({current_input}, {current_output}, mb_id);
      current_input = current_output;
    } catch (const std::exception &e) {
      throw std::runtime_error("Error while forward in layer " + std::to_string(i) + " (" +
                               layers_[i]->name() + "): " + e.what());
    }
  }
  this->device().getFlow(this->flow_handle_)->synchronize();
}

void Sequential::backward_impl(const ConstTensor &grad_output, const Tensor &grad_input,
                               size_t mb_id) {
  if (layers_.empty()) {
    throw std::runtime_error("Cannot backward through empty sequential model");
  }
  ConstTensor current_gradient = grad_output;
  Tensor current_grad_input;
  for (int i = static_cast<int>(layers_.size()) - 1; i >= 0; --i) {
    try {
      // no need to renew buffer since backward doesn't cache inputs
      if (i > 0) {
        current_grad_input = this->get_buffer({max_size_}, grad_output->data_type());
        layers_[i]->backward({current_gradient}, {current_grad_input}, mb_id);
        current_gradient = current_grad_input;
      } else {
        layers_[i]->backward({current_gradient}, {grad_input}, mb_id);
      }
    } catch (const std::exception &e) {
      throw std::runtime_error("Error in backward pass of layer " + std::to_string(i) + " (" +
                               layers_[i]->type() + "): " + e.what());
    }
  }
  this->device().getFlow(this->flow_handle_)->synchronize();
}

Sequential::Sequential(const std::string &name, std::vector<std::unique_ptr<SISOLayer>> layers)
    : Block(name) {
  layers_ = std::move(layers);
}

std::vector<size_t> Sequential::compute_output_shape(const std::vector<size_t> &input_shape) const {
  if (layers_.empty()) {
    return input_shape;
  }

  std::vector<size_t> current_shape = input_shape;
  for (const auto &layer : layers_) {
    current_shape = layer->output_shape({current_shape})[0];
  }

  return current_shape;
}

void Sequential::print_summary(const std::vector<size_t> &input_shape) const {
  if (layers_.empty()) {
    std::cout << "Empty model.\n";
    return;
  }

  auto format_shape = [](const std::vector<size_t> &shape) {
    std::string shape_str = "(";
    for (size_t j = 0; j < shape.size(); ++j) {
      if (j > 0) shape_str += ",";
      shape_str += std::to_string(shape[j]);
    }
    shape_str += ")";
    return shape_str;
  };

  std::cout << std::string(100, '=') << "\n";
  std::cout << "Model Summary: " << name_ << "\n";
  std::cout << std::string(100, '=') << "\n";
  std::cout << std::left << std::setw(20) << "Layer (Type)" << std::setw(20) << "Input Shape"
            << std::setw(20) << "Output Shape" << "\n";

  std::vector<size_t> current_shape = input_shape;
  for (size_t i = 0; i < layers_.size(); ++i) {
    const auto &layer = layers_[i];
    std::cout << std::left << std::setw(20)
              << (layer->get_config().name.empty() ? layer->type() : layer->get_config().name);

    std::cout << std::setw(20) << format_shape(current_shape);

    auto output_shape = layer->output_shape({current_shape})[0];
    std::cout << std::setw(20) << format_shape(output_shape) << "\n";
    current_shape = layer->output_shape({current_shape})[0];
  }
  std::cout << std::string(100, '-') << "\n";
}

std::vector<SISOLayer *> Sequential::get_layers() { return this->layers(); }

LayerConfig Sequential::get_config() const {
  LayerConfig config;
  config.name = name_;
  config.type = TYPE_NAME;
  nlohmann::json layers_config = nlohmann::json::array();
  for (const auto &layer : layers_) {
    auto layer_config = layer->get_config();
    layers_config.push_back(layer_config.to_json());
  }
  config.set("layers", layers_config);
  return config;
}

std::unique_ptr<Sequential> Sequential::create_from_config(const LayerConfig &config) {
  std::vector<std::unique_ptr<SISOLayer>> layers;
  nlohmann::json layers_json = config.get<nlohmann::json>("layers", nlohmann::json::array());
  if (!layers_json.is_array()) {
    throw std::runtime_error("Sequential layer config 'layers' parameter must be an array");
  }
  LayerFactory::register_defaults();
  for (const auto &layer_json : layers_json) {
    LayerConfig layer_config = LayerConfig::from_json(layer_json);
    auto layer = LayerFactory::create(layer_config);
    layers.push_back(std::move(layer));
  }
  return std::make_unique<Sequential>(config.name, std::move(layers));
}

}  // namespace tnn
