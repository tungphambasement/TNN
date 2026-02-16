#pragma once

#include <deque>
#include <vector>

#include "device/iallocator.hpp"
#include "graph_context.hpp"
#include "nn/io_node.hpp"
#include "nn/op_node.hpp"

namespace tnn {

class Graph {
public:
  Graph(IAllocator& allocator, GraphContextDescriptor ctx_desc, std::deque<OpNode>&& op_nodes,
        std::deque<IONode>&& io_nodes, std::vector<OpNode*>&& execution_sequence)
      : ctx_(allocator, ctx_desc),
        op_nodes_(std::move(op_nodes)),
        io_nodes_(std::move(io_nodes)),
        execution_sequence_(std::move(execution_sequence)) {
    for (auto& op_node : op_nodes_) {
      op_node.init(allocator);
    }
  }

  GraphContext& context() { return ctx_; }

  const Device& device() const { return ctx_.device(); }

  const std::vector<OpNode*>& ops() const { return execution_sequence_; }

private:
  friend class GraphExecutor;
  GraphContext ctx_;
  std::deque<OpNode> op_nodes_;
  std::deque<IONode> io_nodes_;
  std::vector<OpNode*> execution_sequence_;
};

}  // namespace tnn