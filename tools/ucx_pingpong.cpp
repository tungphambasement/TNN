
#include <ucp/api/ucp.h>
#include <iostream>
#include <vector>
#include <cstring>
#include <chrono>

static void check(ucs_status_t s, const char* msg) {
  if (s != UCS_OK) { std::cerr << msg << ": " << ucs_status_string(s) << "\n"; exit(1); }
}

int main(int argc, char** argv) {
  bool server = argc > 1 && std::string(argv[1]) == "server";
  size_t bytes = 25690112; // current activation size

  ucp_params_t params{};
  params.field_mask = UCP_PARAM_FIELD_FEATURES;
  params.features = UCP_FEATURE_TAG;

  ucp_config_t* config;
  check(ucp_config_read(nullptr, nullptr, &config), "ucp_config_read");

  ucp_context_h ctx;
  check(ucp_init(&params, config, &ctx), "ucp_init");
  ucp_config_release(config);

  ucp_worker_params_t wparams{};
  wparams.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
  wparams.thread_mode = UCS_THREAD_MODE_SINGLE;

  ucp_worker_h worker;
  check(ucp_worker_create(ctx, &wparams, &worker), "ucp_worker_create");

  // This MVP only proves UCX build/link/runtime is available.
  // Real endpoint exchange will be patched next via TCP bootstrap.
  std::cout << "UCX initialized OK. role=" << (server ? "server" : "client")
            << " test_bytes=" << bytes << "\n";

  ucp_worker_destroy(worker);
  ucp_cleanup(ctx);
  return 0;
}
