#pragma once

#include <vector>

#include "nn/node.hpp"

namespace tnn {

class OpNode;

class IONode : public INode {
public:
  IONode() = default;

  std::string type() const override { return "io_node"; }
  void save_state(std::ofstream& file) override {}
  NodeConfig get_config() const override { return NodeConfig(); }

  void add_producer(OpNode* op_node) { producers_.push_back(op_node); }
  const std::vector<OpNode*>& producers() const { return producers_; }

  void add_consumer(OpNode* op_node) { consumers_.push_back(op_node); }
  const std::vector<OpNode*>& consumers() const { return consumers_; }

private:
  std::vector<OpNode*> producers_;
  std::vector<OpNode*> consumers_;
};

}  // namespace tnn