#pragma once

#include <deque>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "device/iallocator.hpp"
#include "graph_context.hpp"
#include "nn/graph.hpp"
#include "nn/io_node.hpp"
#include "nn/op_node.hpp"
#include "nn/siso_layer.hpp"

namespace tnn {

class GraphBuilder {
public:
  GraphBuilder()
      : ctx_desc_{
            .param_descs = {},
            .param_bytes = 0,
            .grad_bytes = 0,
        } {}

  size_t num_nodes() const { return op_nodes_.size() + io_nodes_.size(); }

  OpNode& add_layer(std::unique_ptr<SISOLayer> siso_layer) {
    std::string uid = "op_" + std::to_string(node_count_++);
    OpNode new_node(uid, ctx_desc_, std::move(siso_layer));
    auto& node = op_nodes_.emplace(new_node.uid(), std::move(new_node)).first->second;
    return node;
  }

  IONode& input(std::string uid = "input") {
    auto node = IONode(uid);
    auto& input_node = io_nodes_.emplace(uid, std::move(node)).first->second;
    return input_node;
  }

  // Assuming simple op node, 1 input, 1 output.
  IONode& output(OpNode& op_node, IONode& input) {
    std::string uid = "io_" + std::to_string(node_count_++);
    auto new_node = IONode(uid);
    auto& output = io_nodes_.emplace(new_node.uid(), std::move(new_node)).first->second;
    add_out(op_node, output);
    add_in(op_node, input);
    return output;
  }

  // topological sort the graph to determine execution order
  void sort() {
    execution_sequence_.clear();
    if (op_nodes_.empty()) return;

    std::unordered_map<OpNode*, int> in_degree;
    for (auto& op_pair : op_nodes_) {
      OpNode& op = op_pair.second;
      in_degree[&op] = 0;
      for (IONode* in_tensor : op.inputs()) {
        // OpNode depends on producers of its inputs
        in_degree[&op] += in_tensor->producers().size();
      }
    }

    std::deque<OpNode*> queue;
    for (auto& op_pair : op_nodes_) {
      OpNode& op = op_pair.second;
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

  // takes a sorted graph and eliminates redundant nodes/layers, e.g. consecutive linear layers can
  // be merged into one
  void reduce() {
    // TODO: implement graph reduction to eliminate redundant nodes/layers
  }

  Graph compile(IAllocator& allocator) {
    sort();
    reduce();
    return Graph(allocator, ctx_desc_, std::move(op_nodes_), std::move(io_nodes_),
                 std::move(execution_sequence_));
  }

private:
  GraphContextDescriptor ctx_desc_;
  std::unordered_map<std::string, IONode> io_nodes_;
  std::unordered_map<std::string, OpNode> op_nodes_;
  std::vector<OpNode*> execution_sequence_;
  size_t node_count_ = 0;

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