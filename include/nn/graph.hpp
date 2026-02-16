#pragma once

#include <deque>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "device/iallocator.hpp"
#include "graph_context.hpp"
#include "nn/io_node.hpp"
#include "nn/op_node.hpp"
#include "nn/siso_layer.hpp"

namespace tnn {

class Graph {
public:
  Graph()
      : ctx_desc_{
            .param_descs = {},
            .param_bytes = 0,
            .grad_bytes = 0,
        } {}

  OpNode& add_layer(SISOLayer& siso_layer) {
    OpNode new_node(ctx_desc_, siso_layer);
    auto& node = op_nodes_.emplace_back(std::move(new_node));
    return node;
  }

  IONode& input() {
    auto node = IONode();
    auto& input_node = io_nodes_.emplace_back(std::move(node));
    return input_node;
  }

  // Assuming simple op node, 1 input, 1 output.
  IONode& output(OpNode& op_node, IONode& input) {
    auto new_node = IONode();
    auto& output = io_nodes_.emplace_back(std::move(new_node));
    add_out(op_node, output);
    add_in(op_node, input);
    return output;
  }

  void compile(IAllocator& allocator) {
    ctx_ = std::make_unique<GraphContext>(ctx_desc_, allocator);
    for (auto& op_node : op_nodes_) {
      op_node.init(allocator);
    }
    sort();
  }

  // topological sort the graph to determine execution order
  void sort() {
    execution_sequence_.clear();
    if (op_nodes_.empty()) return;

    std::unordered_map<OpNode*, int> in_degree;
    for (auto& op : op_nodes_) {
      in_degree[&op] = 0;
      for (IONode* in_tensor : op.inputs()) {
        // OpNode depends on producers of its inputs
        in_degree[&op] += in_tensor->producers().size();
      }
    }

    std::deque<OpNode*> queue;
    for (auto& op : op_nodes_) {
      if (in_degree[&op] == 0) {
        queue.push_back(&op);
      }
    }

    while (!queue.empty()) {
      OpNode* curr = queue.front();
      queue.pop_front();
      execution_sequence_.push_back(curr);

      for (IONode* out_tensor : curr->outputs()) {
        for (OpNode* consumer : out_tensor->consumers()) {
          in_degree[consumer]--;
          if (in_degree[consumer] == 0) {
            queue.push_back(consumer);
          }
        }
      }
    }

    if (execution_sequence_.size() != op_nodes_.size()) {
      throw std::runtime_error("Graph contains a cycle");
    }
  }

  void reduce() {
    // TODO: implement graph reduction to eliminate redundant nodes/layers
  }

  GraphContext& context() {
    if (!ctx_) throw std::runtime_error("Graph not compiled");
    return *ctx_;
  }

  const Device& device() const {
    if (!ctx_) throw std::runtime_error("Graph not compiled");
    return ctx_->device();
  }

  const std::vector<OpNode*>& ops() const { return execution_sequence_; }

private:
  friend class GraphExecutor;
  GraphContextDescriptor ctx_desc_;
  std::unique_ptr<GraphContext> ctx_;
  std::deque<IONode> io_nodes_;
  std::deque<OpNode> op_nodes_;
  std::vector<OpNode*> execution_sequence_;

  void add_in(OpNode& op_node, IONode& input) {
    op_node.add_input(&input);
    input.add_consumer(&op_node);
  }
  void add_out(OpNode& op_node, IONode& output) {
    op_node.add_output(&output);
    output.add_producer(&op_node);
  }
};

}  // namespace tnn