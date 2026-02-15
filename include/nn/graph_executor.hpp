#pragma once

#include "device/iallocator.hpp"
#include "nn/graph.hpp"

namespace tnn {

using InputPack = std::unordered_map<IONode*, Tensor>;
using OutputPack = std::unordered_map<IONode*, Tensor>;

class GraphExecutor {
public:
  GraphExecutor(Graph& graph, IAllocator& allocator)
      : graph_(graph),
        allocator_(allocator) {
    // initialize node outputs
    for (auto& node : graph_.io_nodes_) {
      node_outputs_[&node] = Output{nullptr, nullptr};
    }
  }

  void forward(const InputPack& inputs, OutputPack& outputs) {
    // set input tensors
    for (const auto& [input_node, tensor] : inputs) {
      node_outputs_[input_node].act = tensor;
    }
    // assuming topologically sorted layers, execute forward pass
    for (auto it = graph_.op_nodes_.begin(); it != graph_.op_nodes_.end(); ++it) {
      forward(*it);
    }
    // wire output tensors
    for (auto& [output_node, tensor] : outputs) {
      tensor = node_outputs_[output_node].act;
    }
  }

  void backward(const InputPack& grads, OutputPack& grad_inputs) {
    // set upstream gradients
    for (const auto& [output_node, tensor] : grads) {
      node_outputs_[output_node].grad = tensor;
    }
    for (auto it = graph_.op_nodes_.rbegin(); it != graph_.op_nodes_.rend(); ++it) {
      backward(*it);
    }
    // wire downstream gradients
    for (auto& [input_node, tensor] : grad_inputs) {
      tensor = node_outputs_[input_node].grad;
    }
  }

private:
  Graph& graph_;
  IAllocator& allocator_;

  struct Output {
    Tensor act;
    Tensor grad;
  };

  std::unordered_map<IONode*, Output> node_outputs_;

  void forward(OpNode& node) {
    // gather inputs
    Vec<IONode*>& input_nodes = graph_.ins_[&node];
    Vec<ConstTensor> inputs;
    for (const auto& input_node : input_nodes) {
      const ConstTensor& act = node_outputs_[input_node].act;
      if (!act) {
        throw std::runtime_error("Null input while forwarding graph");
      }
      inputs.push_back(act);
    }

    // gather outputs
    Vec<IONode*>& output_nodes = graph_.outs_[&node];
    Vec<Tensor> outputs;
    auto out_shapes = inputs |
                      std::views::transform([&](const ConstTensor& t) { return t->shape(); }) |
                      std::views::transform(
                          [&](const Vec<size_t>& shape) { return node.output_shape({shape})[0]; });
    DType_t dtype = inputs[0]->data_type();
    if (out_shapes.size() != output_nodes.size()) {
      throw std::runtime_error("Num output shapes does not match num output nodes");
    }
    for (size_t i = 0; i < output_nodes.size(); ++i) {
      IONode* output_node = output_nodes[i];
      Vec<size_t> out_shape = out_shapes[i];
      Tensor& act = node_outputs_[output_node].act;
      if (!act) {
        act = make_tensor(allocator_, dtype, out_shape);
      } else {
        act->ensure(out_shape);
      }
      outputs.push_back(act);
    }

    node.forward(inputs, outputs);
  }

  void backward(OpNode& node) {
    // gather upstream grad_output
    Vec<IONode*>& output_nodes = graph_.outs_[&node];
    Vec<ConstTensor> gradients;
    for (const auto& output_node : output_nodes) {
      Tensor& grad = node_outputs_[output_node].grad;
      if (!grad) {
        throw std::runtime_error("Null output gradient while backwarding graph");
      }
      gradients.push_back(grad);
    }

    // gather downstream grad_output
    Vec<IONode*>& input_nodes = graph_.ins_[&node];
    Vec<Tensor> grad_inputs;
    for (const auto& input_node : input_nodes) {
      Tensor& grad = node_outputs_[input_node].grad;
      // just use the same shape and dtype as input activation, since we don't have the actual
      // input tensor here
      auto& input_act = node_outputs_[input_node].act;
      if (!input_act) {
        throw std::runtime_error("Null input activation while backwarding graph");
      }
      if (!grad) {
        grad = make_tensor(allocator_, input_act->data_type(), input_act->shape());
      } else {
        grad->ensure(input_act->shape());
      }
      grad_inputs.push_back(grad);
    }

    node.backward(gradients, grad_inputs);
  }
};
}  // namespace tnn