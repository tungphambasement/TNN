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
    layers_.push_back(&layer_node);
    return *this;
  }

  Graph& compile() {
    ctx.init();
    for (auto layer : layers_) {
      layer->init();
    }
    return *this;
  }

  GraphContext& context() { return ctx; }

private:
  GraphContext ctx;
  std::vector<Layer*> layers_;
  std::vector<INode*> inputs_;
  std::vector<INode*> outputs_;
  std::vector<INode*> nodes_;
};

}  // namespace tnn