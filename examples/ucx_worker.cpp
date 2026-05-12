#include "distributed/ucx_worker.hpp"

#include <getopt.h>

#include <iostream>
#include <string>

#include "distributed/endpoint.hpp"

using namespace tnn;
using namespace std;

struct Config {
  std::string host = "localhost";
  int port = 0;
  bool use_gpu = false;
};

void print_usage(const char *program_name) {
  cout << "Usage: " << program_name << " [options]" << endl;
  cout << endl;
  cout << "Options:" << endl;
  cout << "  --host <address>       Hostname or IP to bind to (required)" << endl;
  cout << "  --port <number>        TCP port for initial connection (required)" << endl;
  cout << "  --gpu                  Enable GPU offloading" << endl;
  cout << "  -h, --help             Show this help message" << endl;
}

bool parse_arguments(int argc, char *argv[], Config &cfg) {
  int c;

  static struct option long_options[] = {{"host", required_argument, 0, 'H'},
                                         {"port", required_argument, 0, 'p'},                                         {"gpu", no_argument, 0, 'G'},
                                         {"help", no_argument, 0, 'h'},
                                         {0, 0, 0, 0}};

  while ((c = getopt_long(argc, argv, "H:p:Gh", long_options, nullptr)) != -1) {
    switch (c) {
      case 'H':
        cfg.host = optarg;
        break;
      case 'p':
        try {
          cfg.port = stoi(optarg);
        } catch (...) {
          cerr << "Invalid port value: " << optarg << endl;
          return false;
        }
        break;
      case 'G':
        cfg.use_gpu = true;
        break;
      case 'h':
        print_usage(argv[0]);
        return false;
      case '?':
        return false;
      default:
        return false;
    }
  }

  if (cfg.host.empty()) {
    cerr << "Missing required argument: --host" << endl;
    print_usage(argv[0]);
    return false;
  }
  if (cfg.port <= 0 || cfg.port > 65535) {
    cerr << "Invalid or missing port: " << cfg.port << endl;
    print_usage(argv[0]);
    return false;
  }

  return true;
}

int main(int argc, char *argv[]) {
  Config cfg;

  if (!parse_arguments(argc, argv, cfg)) {
    return 1;
  }

  try {
    Endpoint worker_endpoint = Endpoint::ucx(cfg.host, cfg.port);
    UCXWorker worker(worker_endpoint, cfg.use_gpu);
    worker.start();
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
