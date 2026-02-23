#pragma once

#include <unordered_map>
#include <vector>

#include "device/iallocator.hpp"
#include "graph_context.hpp"
#include "nn/blocks_impl/sequential.hpp"
#include "nn/io_node.hpp"
#include "nn/op_node.hpp"
#include "nn/siso_layer.hpp"

namespace tnn {

using GraphConfig = TConfig;

class Graph {
public:
  Graph(IAllocator& allocator, GraphContextDescriptor ctx_desc,
        std::unordered_map<std::string, OpNode>&& op_nodes,
        std::unordered_map<std::string, IONode>&& io_nodes,
        std::vector<OpNode*>&& execution_sequence)
      : ctx_(allocator, ctx_desc),
        op_nodes_(std::move(op_nodes)),
        io_nodes_(std::move(io_nodes)),
        execution_sequence_(std::move(execution_sequence)) {
    for (auto& op_pair : op_nodes_) {
      OpNode& op_node = op_pair.second;
      op_node.init(allocator);
    }
  }

  Graph(const Graph& other) = delete;
  Graph(Graph&& other) = default;
  Graph& operator=(const Graph& other) = delete;
  Graph& operator=(Graph&& other) = delete;

  GraphContext& context() { return ctx_; }

  const Device& device() const { return ctx_.device(); }

  const std::unordered_map<std::string, IONode>& io_nodes() const { return io_nodes_; }
  const IONode& io_node(const std::string& uid) const { return io_nodes_.at(uid); }
  const std::vector<OpNode*>& ops() const { return execution_sequence_; }

  const std::string& name() const { return name_; }
  Graph& set_name(const std::string& name) {
    name_ = name;
    return *this;
  }

  Graph& set_training(bool training) {
    for (auto& [uid, op_node] : op_nodes_) {
      op_node.set_training(training);
    }
    return *this;
  }

  // temporary: get layers from op nodes
  // need to refactor later if we want to support more complex graph structures
  std::vector<SISOLayer*> get_layers() {
    std::vector<SISOLayer*> layers;
    for (auto& op_pair : op_nodes_) {
      OpNode& node = op_pair.second;
      Sequential* seq = dynamic_cast<Sequential*>(node.layer());
      if (seq) {
        auto seq_layers = seq->get_layers();
        layers.insert(layers.end(), seq_layers.begin(), seq_layers.end());
        continue;
      }

      SISOLayer* layer = dynamic_cast<SISOLayer*>(node.layer());
      if (layer) layers.push_back(layer);
    }
    return layers;
  }

  GraphConfig get_config() const {
    GraphConfig config;
    config.type = "graph";
    std::vector<nlohmann::json> op_configs;
    for (const auto& op_pair : op_nodes_) {
      const OpNode& node = op_pair.second;
      op_configs.push_back(node.get_config().to_json());
    }
    config.set("op_nodes", nlohmann::json(op_configs));
    std::vector<nlohmann::json> io_configs;
    for (const auto& io_pair : io_nodes_) {
      const IONode& node = io_pair.second;
      io_configs.push_back(node.get_config().to_json());
    }
    config.set("io_nodes", nlohmann::json(io_configs));
    std::vector<std::string> execution_sequence;
    for (const auto* node : execution_sequence_) {
      execution_sequence.push_back(node->uid());
    }
    config.set("execution_sequence", execution_sequence);
    return config;
  }

private:
  std::string name_;
  GraphContext ctx_;
  std::unordered_map<std::string, OpNode> op_nodes_;
  std::unordered_map<std::string, IONode> io_nodes_;
  std::vector<OpNode*> execution_sequence_;
};

}  // namespace tnn