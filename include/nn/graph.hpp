#pragma once

#include "device/iallocator.hpp"
#include "graph_context.hpp"
#include "nn/layer.hpp"
#include "nn/node.hpp"

namespace tnn {
class Graph {
public:
  Graph(IAllocator& allocator)
      : ctx(allocator) {}

  Graph& add_layer(Layer& layer_node) {
    layer_node.set_context(ctx);
    nodes_.push_back(&layer_node);
    return *this;
  }

  Graph& compile() {
    ctx.init();
    for (auto& node : nodes_) {
      node->init();
    }
    return *this;
  }

  GraphContext& context() { return ctx; }

private:
  GraphContext ctx;
  std::vector<INode*> nodes_;
};

}  // namespace tnn