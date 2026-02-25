/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "nn/blocks_impl/sequential.hpp"

#include <fmt/core.h>

#include <cstddef>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>

#include "nlohmann/json_fwd.hpp"
#include "nn/block.hpp"
#include "nn/layers.hpp"
#include "nn/siso_layer.hpp"
#include "type/type.hpp"

namespace tnn {

// Compute M_i (i in [0, n))
Vec<size_t> Sequential::out_sizes(const std::vector<size_t> &shape, DType_t dtype) {
  Vec<size_t> buffer_size(layers_.size());
  size_t element_size = get_dtype_size(dtype);
  Vec<size_t> current_shape = shape;
  for (size_t i = 0; i < layers_.size(); ++i) {
    current_shape = layers_[i]->output_shape({current_shape})[0];
    buffer_size[i] = std::accumulate(current_shape.begin(), current_shape.end(), element_size,
                                     std::multiplies<size_t>());
  }
  return buffer_size;
}

void Sequential::forward_impl(const ConstTensor &input, const Tensor &output, size_t mb_id) {
  if (layers_.empty()) {
    throw std::runtime_error("Cannot forward through empty sequential model");
  }
  input_shape_cache_[mb_id] = input->shape();
  ConstTensor current_input = input;
  Tensor current_output = nullptr;
  if (is_training_) {
    for (size_t i = 0; i < layers_.size(); ++i) {
      if (i < layers_.size() - 1) {
        current_output = this->get_buffer(layers_[i]->output_shape({current_input->shape()})[0],
                                          input->data_type());
      } else {
        current_output = output;
      }
      layers_[i]->forward({current_input}, {current_output}, mb_id);
      current_input = current_output;
    }
  } else {
    Vec<size_t> m = this->out_sizes(input->shape(), input->data_type());
    size_t m_b = 0;
    for (size_t i = 0; i < m.size() - 1; ++i) {
      m_b = std::max(m_b, m[i] + m[i + 1]);
    }
    dptr buffer = allocator_->allocate(m_b);
    int side = 0;
    DType_t dtype = input->data_type();
    for (size_t i = 0; i < layers_.size(); ++i) {
      if (i < layers_.size() - 1) {
        size_t offset = side ? m_b - m[i] : 0;
        dptr out_ptr = buffer.span(offset, m[i]);
        Vec<size_t> out_shape = layers_[i]->output_shape({current_input->shape()})[0];
        current_output = make_tensor(*allocator_, dtype, out_shape, std::move(out_ptr));
        layers_[i]->forward({current_input}, {current_output}, mb_id);
        side = 1 - side;
        current_input = current_output;
      } else {
        layers_[i]->forward({current_input}, {output}, mb_id);
      }
    }
  }
  this->device().getFlow(this->flow_handle_)->synchronize();
}

void Sequential::backward_impl(const ConstTensor &grad_output, const Tensor &grad_input,
                               size_t mb_id) {
  if (layers_.empty()) {
    throw std::runtime_error("Cannot backward through empty sequential model");
  }
  auto it_input_shape = input_shape_cache_.find(mb_id);
  if (it_input_shape == input_shape_cache_.end()) {
    throw std::runtime_error("No cached input shape found for micro-batch ID: " +
                             std::to_string(mb_id));
  }
  Vec<size_t> input_shape = it_input_shape->second;
  Vec<size_t> m = this->out_sizes(input_shape, grad_output->data_type());
  Vec<Vec<size_t>> out_shapes(layers_.size());
  out_shapes[0] = layers_[0]->output_shape({input_shape})[0];
  for (size_t i = 1; i < layers_.size(); ++i) {
    out_shapes[i] = layers_[i]->output_shape({out_shapes[i - 1]})[0];
  }
  size_t m_b = 0;
  for (size_t i = 0; i < m.size() - 1; ++i) {
    m_b = std::max(m_b, m[i] + m[i + 1]);
  }
  dptr buffer = allocator_->allocate(m_b);
  ConstTensor current_gradient = grad_output;
  Tensor current_grad_input = nullptr;
  int side = 0;
  for (int i = static_cast<int>(layers_.size()) - 1; i >= 0; --i) {
    // no need to renew buffer since backward doesn't cache inputs
    if (i > 0) {
      size_t offset = side ? m_b - m[i] : 0;
      Vec<size_t> grad_input_shape = out_shapes[i - 1];
      dptr grad_input_ptr = buffer.span(offset, m[i]);
      current_grad_input = make_tensor(*allocator_, grad_output->data_type(), grad_input_shape,
                                       std::move(grad_input_ptr));
      layers_[i]->backward({current_gradient}, {current_grad_input}, mb_id);
      side = 1 - side;
      current_gradient = current_grad_input;
    } else {
      layers_[i]->backward({current_gradient}, {grad_input}, mb_id);
    }
  }
  this->device().getFlow(this->flow_handle_)->synchronize();
}

Sequential::Sequential(std::vector<std::unique_ptr<SISOLayer>> layers, const std::string &name)
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
  return std::make_unique<Sequential>(std::move(layers), config.name);
}

}  // namespace tnn
