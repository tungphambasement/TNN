#pragma once

#include <deque>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "device/iallocator.hpp"
#include "graph_context.hpp"
#include "nn/io_node.hpp"
#include "nn/layer.hpp"
#include "nn/op_node.hpp"
#include "tensor/tensor.hpp"
#include "tensor/tensor_factory.hpp"
#include "type/type.hpp"

namespace tnn {

class Graph {
public:
  Graph(IAllocator& allocator)
      : ctx_(allocator) {}

  OpNode& add_layer(Layer& layer_node) {
    OpNode new_node(ctx_, layer_node);
    auto& node = op_nodes_.emplace_back(std::move(new_node));
    return node;
  }

  IONode& input() {
    auto node = IONode(ctx_);
    auto& input_node = io_nodes_.emplace_back(std::move(node));
    return input_node;
  }

  // Assuming simple op node, 1 input, 1 output.
  IONode& output(OpNode& op_node, IONode& input) {
    auto new_node = IONode(ctx_);
    auto& output = io_nodes_.emplace_back(std::move(new_node));
    add_out(op_node, output);
    add_in(op_node, input);
    return output;
  }

  Graph& compile() {
    ctx_.init();
    for (auto& op_node : op_nodes_) {
      op_node.init();
    }
    sort();  // hope it works
    return *this;
  }

  // topological sort the graph to determine execution order
  // interestingly, topologically sorted graphs when having their edge reversed, are sorted in exact
  // reverse order. This means we can do backward pass by simply iterating through the same sorted
  // list in reverse.
  void sort() {
    if (op_nodes_.empty()) return;
    std::unordered_map<OpNode*, int> in_degree;
    for (auto& op : op_nodes_) {
      in_degree[&op] = 0;
      for (IONode* in_tensor : ins_[&op]) {
        // If an IONode is produced by another OpNode, it's a dependency
        if (producers_.count(in_tensor)) {
          in_degree[&op] += producers_[in_tensor].size();
        }
      }
    }

    std::deque<OpNode*> queue;
    for (auto& op : op_nodes_) {
      if (in_degree[&op] == 0) {
        queue.push_back(&op);
      }
    }

    Vec<OpNode*> sorted_ptrs;
    while (!queue.empty()) {
      OpNode* curr = queue.front();
      queue.pop_front();
      sorted_ptrs.push_back(curr);

      for (IONode* out_tensor : outs_[curr]) {
        for (OpNode* consumer : consumers_[out_tensor]) {
          if (--in_degree[consumer] == 0) {
            queue.push_back(consumer);
          }
        }
      }
    }

    if (sorted_ptrs.size() != op_nodes_.size()) {
      throw std::runtime_error("Graph contains a cycle");
    }

    reorder_and_remap(sorted_ptrs);
  }

  void reduce() {
    // TODO: implement graph reduction to eliminate redundant nodes/layers
  }

  GraphContext& context() { return ctx_; }

private:
  friend class GraphExecutor;
  GraphContext ctx_;
  std::deque<IONode> io_nodes_;
  std::deque<OpNode> op_nodes_;
  std::unordered_map<OpNode*, Vec<IONode*>> ins_;
  std::unordered_map<OpNode*, Vec<IONode*>> outs_;
  std::unordered_map<IONode*, Vec<OpNode*>> consumers_;
  std::unordered_map<IONode*, Vec<OpNode*>> producers_;

  void add_in(OpNode& op_node, IONode& input) {
    ins_[&op_node].push_back(&input);
    consumers_[&input].push_back(&op_node);
  }
  void add_out(OpNode& op_node, IONode& output) {
    outs_[&op_node].push_back(&output);
    producers_[&output].push_back(&op_node);
  }

  void reorder_and_remap(const Vec<OpNode*>& sorted_ptrs) {
    std::deque<OpNode> sorted_nodes;
    std::unordered_map<OpNode*, OpNode*> old_to_new;
    for (OpNode* old_ptr : sorted_ptrs) {
      sorted_nodes.push_back(std::move(*old_ptr));
      old_to_new[old_ptr] = &sorted_nodes.back();
    }
    auto remap_map = [&](std::unordered_map<OpNode*, Vec<IONode*>>& map) {
      std::unordered_map<OpNode*, Vec<IONode*>> next_map;
      for (auto& [old_key, val] : map) {
        next_map[old_to_new[old_key]] = std::move(val);
      }
      map = std::move(next_map);
    };
    remap_map(ins_);
    remap_map(outs_);
    for (auto& [io, ops] : producers_) {
      for (auto& ptr : ops) ptr = old_to_new[ptr];
    }
    for (auto& [io, ops] : consumers_) {
      for (auto& ptr : ops) ptr = old_to_new[ptr];
    }
    op_nodes_ = std::move(sorted_nodes);
  }
};

using InputPack = std::unordered_map<IONode*, Tensor>;
using OutputPack = std::unordered_map<IONode*, Tensor>;

class GraphExecutor {
public:
  GraphExecutor(Graph& graph)
      : graph_(graph) {
    // initialize node outputs
    for (auto& node : graph_.io_nodes_) {
      node_outputs_[&node] = Output{nullptr, nullptr};
    }
  }

  void forward(InputPack inputs, OutputPack outputs) {
    // set input tensors
    for (auto& [input_node, tensor] : inputs) {
      node_outputs_[input_node].act = tensor;
    }
    // gather output tensors
    for (auto& [output_node, tensor] : outputs) {
      node_outputs_[output_node].act = tensor;
    }
    // assuming topologically sorted layers, execute forward pass
    for (auto it = graph_.op_nodes_.begin(); it != graph_.op_nodes_.end(); ++it) {
      forward(*it);
    }
  }

  void backward(InputPack grads, OutputPack grad_inputs) {
    // set upstream gradients
    for (auto& [output_node, tensor] : grads) {
      node_outputs_[output_node].grad = tensor;
    }
    // gather downstream gradients
    for (auto& [input_node, tensor] : grad_inputs) {
      node_outputs_[input_node].grad = tensor;
    }
    for (auto it = graph_.op_nodes_.rbegin(); it != graph_.op_nodes_.rend(); ++it) {
      backward(*it);
    }
  }

private:
  Graph& graph_;

  struct Output {
    Tensor act;
    Tensor grad;
  };

  std::unordered_map<IONode*, Output> node_outputs_;

  void forward(OpNode& node) {
    // gather inputs
    Vec<IONode*>& input_nodes = graph_.ins_[&node];
    Vec<ConstTensor> inputs;
    for (auto& input_node : input_nodes) {
      Tensor& act = node_outputs_[input_node].act;
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
        act = make_tensor(graph_.context().allocator(), dtype, out_shape);
      } else {
        act->ensure(out_shape);
      }
      outputs.push_back(act);
    }

    node.forward(inputs, outputs);
  }

  void backward(OpNode& node) {
    // gather upstream gradient
    Vec<IONode*>& output_nodes = graph_.outs_[&node];
    Vec<ConstTensor> gradients;
    for (auto& output_node : output_nodes) {
      Tensor& grad = node_outputs_[output_node].grad;
      gradients.push_back(grad);
    }

    // gather downstream gradient
    Vec<IONode*>& input_nodes = graph_.ins_[&node];
    Vec<Tensor> grad_inputs;
    for (auto& input_node : input_nodes) {
      Tensor& grad = node_outputs_[input_node].grad;
      // just use the same shape and dtype as input activation, since we don't have the actual
      // input tensor here
      auto& input_act = node_outputs_[input_node].act;
      if (!input_act) {
        throw std::runtime_error("Null input activation while backwarding graph");
      }
      if (!grad) {
        grad =
            make_tensor(graph_.context().allocator(), input_act->data_type(), input_act->shape());
      } else {
        grad->ensure(input_act->shape());
      }
      grad_inputs.push_back(grad);
    }

    node.backward(gradients, grad_inputs);
  }
};

}  // namespace tnn