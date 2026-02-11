#pragma once

#include "nn/graph_context.hpp"
#include "nn/node.hpp"

namespace tnn {
class IONode : public INode {
public:
  IONode(GraphContext& ctx)
      : INode(ctx) {}

  std::string type() const override { return "io_node"; }
  void save_state(std::ofstream& file) override {}
  NodeConfig get_config() const override { return NodeConfig(); }
};

}  // namespace tnn