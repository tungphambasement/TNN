#include "nn/graph.hpp"

#include <getopt.h>

#include <unordered_map>

#include "data_loading/data_loader_factory.hpp"
#include "device/device_manager.hpp"
#include "device/device_type.hpp"
#include "nn/accuracy.hpp"
#include "nn/example_models.hpp"
#include "nn/graph_executor.hpp"
#include "nn/io_node.hpp"
#include "nn/layers.hpp"
#include "nn/train.hpp"
#include "utils/env.hpp"

using namespace std;
using namespace tnn;

signed main(int argc, char *argv[]) {
  ExampleModels::register_defaults();

  std::string config_path;
  static struct option long_options[] = {
      {"config", required_argument, 0, 'c'}, {"help", no_argument, 0, 'h'}, {0, 0, 0, 0}};

  int opt;
  while ((opt = getopt_long(argc, argv, "c:h", long_options, nullptr)) != -1) {
    switch (opt) {
      case 'c':
        config_path = optarg;
        break;
      case 'h':
        cout << "Usage: " << argv[0] << " [options]" << endl;
        cout << "Options:" << endl;
        cout << "  --config <path>    Path to the JSON configuration file" << endl;
        cout << "  -h, --help         Show this help message" << endl;
        return 0;
      default:
        return 1;
    }
  }

  TrainingConfig train_config;
  train_config.load_from_json(config_path);
  train_config.print_config();

  const Device &device = train_config.device_type == DeviceType::GPU ? getGPU() : getHost();
  auto &allocator = PoolAllocator::instance(device, defaultFlowHandle);
  Graph graph;

  auto model = ExampleModels::create(train_config.model_name);
  auto layers = model.get_layers();
  std::string input_name = "input";
  std::string output_name = layers.back()->name();

  std::unordered_map<std::string, IONode *> nodes;
  nodes[input_name] = &graph.input();
  IONode *current_input = nodes[input_name];
  for (const auto &layer : layers) {
    auto &node = graph.add_layer(*layer);
    auto &output = graph.output(node, *current_input);
    nodes[layer->name()] = &output;
    current_input = &output;
  }

  graph.compile(allocator);
  GraphExecutor executor(graph, allocator);

  auto [train_loader, val_loader] =
      DataLoaderFactory::create(train_config.dataset_name, train_config.dataset_path);
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
        {nodes[input_name], device_input},
    };
    OutputPack outputs = {
        {nodes[output_name], device_output},
    };
    executor.forward(inputs, outputs);
    device_output = outputs[nodes[output_name]];
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
        {nodes[output_name], grad_output},
    };
    OutputPack grad_inputs = {
        {nodes[input_name], grad_input},
    };
    executor.backward(grad_outputs, grad_inputs);
    grad_input = grad_inputs[nodes[input_name]];

    optimizer->update();
    optimizer->clear_gradients();
  }

  return 0;
}