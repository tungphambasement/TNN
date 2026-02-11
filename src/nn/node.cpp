/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "nn/node.hpp"

namespace tnn {

INode::INode()
    : context_(nullptr) {}

void INode::set_context(GraphContext &graph_ctx) {
  context_ = &graph_ctx;
  on_set_context(graph_ctx);
}

GraphContext &INode::context() const {
  if (!context_) {
    throw std::runtime_error("Context is not set");
  }
  return *context_;
}

}  // namespace tnn
