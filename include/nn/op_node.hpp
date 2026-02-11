#pragma once

#include "nn/graph_context.hpp"
#include "nn/layer.hpp"
#include "nn/node.hpp"

namespace tnn {

// Simple op node. 1 Input, 1 Output.
class OpNode : public INode {
public:
  OpNode(GraphContext &context, Layer &layer)
      : INode(context),
        layer_(layer) {
    auto param_descriptors = layer_.param_descriptors();
    for (const auto &desc : param_descriptors) {
      context_.register_param(desc.shape, desc.dtype);
    }
    layer_.set_allocator(context_.allocator());
  }

  void init() {
    auto param_descriptors = layer_.param_descriptors();
    for (const auto &desc : param_descriptors) {
      *desc.data_ptr = context_.get_param(desc.shape, desc.dtype);
      *desc.grad_ptr = context_.get_grad(desc.shape, desc.dtype);
    }
    layer_.init();
  }

  void forward(const std::vector<ConstTensor> &inputs, const std::vector<Tensor> &outputs,
               size_t mb_id = 0) {
    layer_.forward(inputs, outputs, mb_id);
  }

  void backward(const std::vector<ConstTensor> &gradients, const std::vector<Tensor> &grad_inputs,
                size_t mb_id = 0) {
    layer_.backward(gradients, grad_inputs, mb_id);
  }

  Vec<Vec<size_t>> output_shape(const Vec<Vec<size_t>> &input_shapes) const {
    return layer_.output_shape(input_shapes);
  }

  std::string type() const override { return "op_node"; }
  void save_state(std::ofstream &file) override {}
  NodeConfig get_config() const override { return NodeConfig(); }

private:
  Layer &layer_;
};

}  // namespace tnn