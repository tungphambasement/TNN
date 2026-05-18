#pragma once

#include <initializer_list>
#include <string>

#include "nn/layer.hpp"
#include "type/type.hpp"

namespace tnn {
class IONode;
class Edge;

using IONodePtr = std::shared_ptr<IONode>;
using EdgePtr = std::shared_ptr<Edge>;

class IONode {
public:
  explicit IONode() = default;

  EdgePtr producer() { return producer_; }
  const EdgePtr producer() const { return producer_; }
  void set_producer(const EdgePtr &producer) { producer_ = producer; }

  void set_name(const std::string &name) { name_ = name; }
  std::string name() const { return name_; }

  const Tensor &data() const { return data_; }
  void set_data(const Tensor &data) { data_ = data; }

  const Tensor &grad() const { return grad_; }
  void set_grad(const Tensor &grad) { grad_ = grad; }

private:
  EdgePtr producer_;
  std::string name_;
  Tensor data_;
  Tensor grad_;
};

class Edge {
public:
  Edge(Layer *layer, const Vec<IONodePtr> &inputs, const Vec<IONodePtr> &outputs)
      : layer_(layer),
        producers_(inputs),
        consumers_(outputs) {}

  Edge(Layer *layer, std::initializer_list<IONodePtr> inputs,
       std::initializer_list<IONodePtr> outputs)
      : layer_(layer),
        producers_(inputs),
        consumers_(outputs) {}

  void accumulate() {
    Vec<ConstTensor> input_data;
    for (const auto &producer : producers_) {
      // Accumulate data from producer to layer input
      if (!producer->data()) {
        producer->producer()->accumulate();
      }
      input_data.push_back(producer->data());
    }

    Vec<Tensor> output_data = layer_->forward(input_data);

    for (size_t i = 0; i < consumers_.size(); ++i) {
      consumers_[i]->set_data(output_data[i]);
    }
  }

private:
  Layer *layer_;
  Vec<IONodePtr> producers_;
  Vec<IONodePtr> consumers_;
};

}  // namespace tnn