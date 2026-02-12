

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
                    .dense(10, "dense")
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

  // Tensor input = make_tensor<float>({64, 28, 28, 1});
  Tensor input, output, label, grad_output, grad_input;
  train_loader->get_batch(64, input, label);
  auto criterion = LossFactory::create_logsoftmax_crossentropy();
  auto optimizer =
      OptimizerFactory::create_adam(train_config.lr_initial, 0.9f, 0.999f, 10e-4f, 3e-4f, false);

  optimizer->attach(graph.context());

  while (train_loader->get_batch(256, input, label)) {
    InputPack inputs = {
        {nodes["image"], input},
    };
    OutputPack outputs = {
        {nodes["dense"], output},
    };
    executor.forward(inputs, outputs);
    float loss;
    criterion->compute_loss(output, label, loss);
    int class_corrects = compute_class_corrects(output, label);
    std::cout << "Loss: " << loss
              << ", Accuracy: " << (static_cast<float>(class_corrects) / 256) * 100.0f << "%"
              << std::endl;
    criterion->compute_gradient(output, label, grad_output);
    InputPack grad_outputs = {
        {nodes["dense"], grad_output},
    };
    OutputPack grad_inputs = {
        {nodes["image"], grad_input},
    };
    executor.backward(grad_outputs, grad_inputs);

    optimizer->update();
    optimizer->clear_gradients();
  }

  return 0;
}