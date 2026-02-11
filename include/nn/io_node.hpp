#pragma once

#include "nn/node.hpp"

namespace tnn {
class IONode : public INode {
public:
  IONode() = default;

  const Tensor &tensor() const {
    if (!tensor_) {
      throw std::runtime_error("Dangling tensor in IO Node");
    }
    return *tensor_;
  }

  Tensor &tensor() {
    if (!tensor_) {
      throw std::runtime_error("Dangling tensor in IO Node");
    }
    return *tensor_;
  }

private:
  Tensor *tensor_;
};

}  // namespace tnn