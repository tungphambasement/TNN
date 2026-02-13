

#include "nn/graph.hpp"

#include <unordered_map>

#include "data_loading/data_loader_factory.hpp"
#include "device/device_manager.hpp"
#include "device/device_type.hpp"
#include "nn/accuracy.hpp"
#include "nn/example_models.hpp"
#include "nn/io_node.hpp"
#include "nn/layers.hpp"
#include "nn/train.hpp"
#include "type/type.hpp"
#include "utils/env.hpp"

using namespace std;
using namespace tnn;

signed main() {
  ExampleModels::register_defaults();

  TrainingConfig train_config;
  train_config.load_from_env();
  train_config.print_config();

  const Device &device = train_config.device_type == DeviceType::GPU ? getGPU() : getCPU();
  auto &allocator = PoolAllocator::instance(device, defaultFlowHandle);
  Graph graph(allocator);

  auto layers = LayerBuilder()
                    .input({28, 28, 1})
                    .conv2d(16, 3, 3, 1, 1, 1, 1, true, "conv1")
                    .batchnorm(16, 1e-5, true, SBool::TRUE, "bn1")
                    .flatten(1, -1, "flatten")
                    .dense(10, true, "dense")
                    .build();

  std::unordered_map<std::string, IONode *> nodes;
  nodes["image"] = &graph.input();
  IONode *current_input = nodes["image"];
  for (const auto &layer : layers) {
    auto &node = graph.add_layer(*layer);
    auto &output = graph.output(node, *current_input);
    nodes[layer->name()] = &output;
    current_input = &output;
  }

  graph.compile();
  GraphExecutor executor(graph);

  string dataset_name = Env::get<std::string>("DATASET_NAME", "");
  string dataset_path = Env::get<std::string>("DATASET_PATH", "data");
  auto [train_loader, val_loader] = DataLoaderFactory::create(dataset_name, dataset_path);
  train_loader->set_seed(123456);

  Tensor input, label;
  auto criterion = LossFactory::create_logsoftmax_crossentropy();
  auto optimizer =
      OptimizerFactory::create_adam(train_config.lr_initial, 0.9f, 0.999f, 10e-4f, 3e-4f, false);

  optimizer->attach(graph.context());

  while (train_loader->get_batch(256, input, label)) {
    Tensor device_input = input->to_device(graph.context().device());
    Tensor device_output;
    InputPack inputs = {
        {nodes["image"], device_input},
    };
    OutputPack outputs = {
        {nodes["dense"], device_output},
    };
    executor.forward(inputs, outputs);
    device_output = outputs[nodes["dense"]];
    float loss;
    Tensor device_labels = label->to_device(graph.context().device());
    criterion->compute_loss(device_output, device_labels, loss);
    int class_corrects = compute_class_corrects(device_output, device_labels);
    std::cout << "Loss: " << loss << ", Accuracy: "
              << (static_cast<float>(class_corrects) / device_output->dimension(0)) * 100.0f << "%"
              << std::endl;
    Tensor grad_output = create_like(device_output), grad_input = create_like(device_input);
    criterion->compute_gradient(device_output, device_labels, grad_output);
    InputPack grad_outputs = {
        {nodes["dense"], grad_output},
    };
    OutputPack grad_inputs = {
        {nodes["image"], grad_input},
    };
    executor.backward(grad_outputs, grad_inputs);
    grad_input = grad_inputs[nodes["image"]];

    optimizer->update();
    optimizer->clear_gradients();
  }

  return 0;
}