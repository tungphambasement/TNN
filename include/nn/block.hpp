#pragma once

#include "nn/layer.hpp"

namespace tnn {
class Block : public Layer {
public:
  Block(const std::string &name = "block") { this->name_ = name; }

protected:
  void register_impl() override {}
  void init_impl() override = 0;
  void on_set_context(GraphContext &context) override = 0;
  void on_set_flow_handle(flowHandle_t handle) override = 0;
  void on_set_seed(unsigned long long seed) override {}
  void on_set_training(bool training) override = 0;
  void on_set_io_dtype(DType_t dtype) override = 0;
  void on_set_param_dtype(DType_t dtype) override = 0;
  void on_set_compute_dtype(DType_t dtype) override = 0;
};
}  // namespace tnn