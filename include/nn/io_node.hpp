#pragma once

#include "nn/node.hpp"

namespace tnn {
class IONode : public INode {
public:
  IONode() = default;

  std::string type() const override { return "io_node"; }
  void save_state(std::ofstream& file) override {}
  NodeConfig get_config() const override { return NodeConfig(); }
};

}  // namespace tnn