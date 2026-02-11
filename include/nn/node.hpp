#pragma once

#include "common/config.hpp"
#include "nn/graph_context.hpp"

namespace tnn {

using NodeConfig = TConfig;

class INode {
public:
  INode(GraphContext &ctx)
      : context_(ctx) {}

  virtual ~INode() = default;

  /**
   * @brief Initialize the layer (e.g., allocate parameters)
   * ! Must set io, param, compute dtypes and device ptr prior to this to ensure proper math.
   * ! Must be called before forward/backward.
   */
  virtual std::string type() const = 0;
  virtual void save_state(std::ofstream &file) = 0;
  virtual NodeConfig get_config() const = 0;

protected:
  GraphContext &context_;
};
}  // namespace tnn