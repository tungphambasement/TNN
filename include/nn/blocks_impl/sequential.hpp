/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <fmt/core.h>

#include <cstddef>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "nn/block.hpp"
#include "nn/layer.hpp"
#include "tensor/tensor.hpp"

namespace tnn {
class Sequential : public Block {
private:
  std::vector<std::unique_ptr<Layer>> layers_;
  size_t max_size_ = 0;

  void compute_max_size(const std::vector<size_t> &input_shape, DType_t dtype);

protected:
  std::vector<Layer *> layers() override {
    std::vector<Layer *> layers;
    for (auto &layer : layers_) {
      layers.push_back(layer.get());
    }
    return layers;
  }
  void init_impl() override;
  void forward_impl(const ConstTensor &input, const Tensor &output, size_t mb_id = 0) override;
  void backward_impl(const ConstTensor &gradient, const Tensor &grad_input,
                     size_t mb_id = 0) override;

public:
  explicit Sequential(const std::string &name = "seq",
                      std::vector<std::unique_ptr<Layer>> layers = {});

  static constexpr const char *TYPE_NAME = "sequential";

  /**
   * @brief Returns the output shape for given input shape
   * @param input_shape The shape of the input tensor as a vector of sizes.
   */
  std::vector<size_t> compute_output_shape(const std::vector<size_t> &input_shape) const override;
  void print_summary(const std::vector<size_t> &input_shape) const;
  const std::vector<Layer *> &get_layers() const;
  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;
  static std::unique_ptr<Sequential> create_from_config(const LayerConfig &config);
};

}  // namespace tnn