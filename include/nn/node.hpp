#pragma once

#include "common/config.hpp"
#include "device/device.hpp"
#include "nn/graph_context.hpp"

namespace tnn {

using NodeConfig = TConfig;

class INode {
public:
  INode();
  virtual ~INode() = default;

  const Device &device() const { return context_->device(); }

  void set_context(GraphContext &graph_ctx);
  GraphContext &context() const;

  /**
   * @brief Initialize the layer (e.g., allocate parameters)
   * ! Must set io, param, compute dtypes and device ptr prior to this to ensure proper math.
   * ! Must be called before forward/backward.
   */
  void init();
  virtual std::string type() const = 0;
  virtual void save_state(std::ofstream &file) = 0;
  virtual NodeConfig get_config() const = 0;

protected:
  bool initialized_ = false;
  GraphContext *context_;

  virtual void on_set_context(GraphContext &graph_ctx) {}
  virtual void on_set_flow_handle(flowHandle_t handle) {}
  virtual void register_impl() = 0;
  virtual void init_impl() = 0;
};
}  // namespace tnn