#pragma once

#include "distributed/coordinator.hpp"
#include "distributed/ucx_communicator.hpp"

namespace tnn {

class UCXCoordinator : public Coordinator {
public:
  UCXCoordinator(CoordinatorConfig config)
      : Coordinator(std::move(config)) {

    auto communicator =
        UCXCommunicator::create(this->coordinator_endpoint_,
                                UCXCommunicator::Config{4});

    this->comm_ = std::move(communicator);
    this->add_message_callback();
  }

  ~UCXCoordinator() = default;
};

} // namespace tnn
