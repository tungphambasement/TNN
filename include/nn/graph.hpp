#pragma once

#include <deque>
#include <stdexcept>
#include <unordered_map>

#include "device/iallocator.hpp"
#include "graph_context.hpp"
#include "nn/io_node.hpp"
#include "nn/layer.hpp"
#include "nn/op_node.hpp"
#include "type/type.hpp"

namespace tnn {

class Graph {
public:
  Graph()
      : ctx_desc_{
            .param_descs = {},
            .param_bytes = 0,
            .grad_bytes = 0,
        } {}

  OpNode& add_layer(Layer& layer_node) {
    OpNode new_node(ctx_desc_, layer_node);
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
    sort();  // hope it works
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

  GraphContext& context() {
    if (!ctx_) throw std::runtime_error("Graph not compiled");
    return *ctx_;
  }

  const Device& device() const {
    if (!ctx_) throw std::runtime_error("Graph not compiled");
    return ctx_->device();
  }

private:
  friend class GraphExecutor;
  GraphContextDescriptor ctx_desc_;
  std::unique_ptr<GraphContext> ctx_;
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

}  // namespace tnn