#include <getopt.h>

#include <memory>

#include "data_loading/data_loader_factory.hpp"
#include "device/device_manager.hpp"
#include "nn/blocks_impl/sequential.hpp"
#include "nn/example_models.hpp"
#include "nn/graph.hpp"
#include "nn/layers.hpp"
#include "nn/schedulers.hpp"
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

  // Load config from JSON file if --config or CONFIG_PATH is set, otherwise use environment
  // variables
  if (config_path.empty()) {
    config_path = Env::get<std::string>("CONFIG_PATH", "");
  }

  if (!config_path.empty()) {
    train_config.load_from_json(config_path);
  } else {
    train_config.load_from_env();
  }
  train_config.print_config();

  // Prioritize loading existing model, else create from available ones
  const auto &device = DeviceManager::getInstance().getDevice(train_config.device_type);
  auto &allocator = PoolAllocator::instance(device, defaultFlowHandle);
  Graph graph;

  string dataset_name = train_config.dataset_name;
  if (dataset_name.empty()) {
    throw std::runtime_error("DATASET_NAME environment variable is not set!");
  }
  string dataset_path = train_config.dataset_path;
  auto [train_loader, val_loader] = DataLoaderFactory::create(dataset_name, dataset_path);
  if (!train_loader || !val_loader) {
    cerr << "Failed to create data loaders for dataset: " << train_config.dataset_name << endl;
    return 1;
  }
  train_loader->set_seed(123456);

  std::unique_ptr<Sequential> model;
  if (!train_config.model_path.empty()) {
    cout << "Loading model from: " << train_config.model_path << endl;
    std::ifstream file(train_config.model_path, std::ios::binary);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to open model file");
    }
    model = load_state<Sequential>(file, graph, allocator);
    file.close();
  } else {
    cout << "Creating model: " << train_config.model_name << endl;
    try {
      Sequential temp_model = ExampleModels::create(train_config.model_name);
      model = std::make_unique<Sequential>(std::move(temp_model));
    } catch (const std::exception &e) {
      cerr << "Error creating model: " << e.what() << endl;
      cout << "Available models are: ";
      for (const auto &name : ExampleModels::available_models()) {
        cout << name << "\n";
      }
      cout << endl;
      return 1;
    }
    model->set_seed(123456);
    graph.add_layer(*model);
    graph.compile(allocator);
  }

  cout << "Training model on device: "
       << (train_config.device_type == DeviceType::CPU ? "CPU" : "GPU") << endl;

  auto criterion = LossFactory::create_logsoftmax_crossentropy();
  auto optimizer =
      OptimizerFactory::create_adam(train_config.lr_initial, 0.9f, 0.999f, 10e-4f, 3e-4f, false);
  auto scheduler = SchedulerFactory::create_step_lr(
      optimizer.get(), 5 * train_loader->size() / train_config.batch_size, 0.1f);

  try {
    train_model(model, graph.context(), train_loader, val_loader, optimizer, criterion, scheduler,
                train_config);
  } catch (const std::exception &e) {
    cerr << "Training failed: " << e.what() << endl;
    return 1;
  }

  return 0;
}