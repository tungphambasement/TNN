#include <ucp/api/ucp.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

struct ReqCtx {
  bool done = false;
  ucs_status_t status = UCS_OK;
};

static void die(const std::string& s) {
  std::cerr << s << "\n";
  std::exit(1);
}

static void check(ucs_status_t s, const char* msg) {
  if (s != UCS_OK) die(std::string(msg) + ": " + ucs_status_string(s));
}

static void send_all(int fd, const void* p, size_t n) {
  const char* c = (const char*)p;
  while (n) {
    ssize_t r = ::send(fd, c, n, 0);
    if (r <= 0) die("tcp send failed");
    c += r;
    n -= r;
  }
}

static void recv_all(int fd, void* p, size_t n) {
  char* c = (char*)p;
  while (n) {
    ssize_t r = ::recv(fd, c, n, MSG_WAITALL);
    if (r <= 0) die("tcp recv failed");
    c += r;
    n -= r;
  }
}

static int listen_one(int port) {
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  int yes = 1;
  setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

  sockaddr_in a{};
  a.sin_family = AF_INET;
  a.sin_addr.s_addr = INADDR_ANY;
  a.sin_port = htons(port);

  if (bind(fd, (sockaddr*)&a, sizeof(a)) != 0) die("bind failed");
  if (listen(fd, 1) != 0) die("listen failed");

  std::cout << "[TCP bootstrap] listening on port " << port << "\n";
  int c = accept(fd, nullptr, nullptr);
  if (c < 0) die("accept failed");
  close(fd);
  return c;
}

static int connect_to(const std::string& host, int port) {
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  sockaddr_in a{};
  a.sin_family = AF_INET;
  a.sin_port = htons(port);
  if (inet_pton(AF_INET, host.c_str(), &a.sin_addr) != 1) die("bad host ip");

  while (connect(fd, (sockaddr*)&a, sizeof(a)) != 0) {
    usleep(200000);
  }
  return fd;
}

static void send_cb(void* request, ucs_status_t status) {
  auto* ctx = (ReqCtx*)request;
  ctx->status = status;
  ctx->done = true;
}

static void recv_cb(void* request, ucs_status_t status, ucp_tag_recv_info_t*) {
  auto* ctx = (ReqCtx*)request;
  ctx->status = status;
  ctx->done = true;
}

static void wait_req(ucp_worker_h worker, void* req, ReqCtx& ctx) {
  if (req == nullptr) return;
  if (UCS_PTR_IS_ERR(req)) {
    die(std::string("UCX request error: ") + ucs_status_string(UCS_PTR_STATUS(req)));
  }
  while (!ctx.done) {
    ucp_worker_progress(worker);
  }
  check(ctx.status, "UCX op");
  ucp_request_free(req);
}

static void exchange_addr(int fd, const void* my_addr, size_t my_len,
                          std::vector<char>& peer_addr) {
  uint64_t n = my_len;
  send_all(fd, &n, sizeof(n));
  send_all(fd, my_addr, my_len);

  uint64_t peer_n = 0;
  recv_all(fd, &peer_n, sizeof(peer_n));
  peer_addr.resize(peer_n);
  recv_all(fd, peer_addr.data(), peer_n);
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr
      << "Usage:\n"
      << "  server: " << argv[0] << " server [port] [iters] [bytes]\n"
      << "  client: " << argv[0] << " client <server_ip> [port] [iters] [bytes]\n";
    return 1;
  }

  std::string role = argv[1];
  bool is_server = role == "server";
  bool is_client = role == "client";
  if (!is_server && !is_client) die("role must be server or client");

  std::string host = is_client ? argv[2] : "";
  int argbase = is_client ? 3 : 2;

  int port = argc > argbase ? std::stoi(argv[argbase]) : 18515;
  int iters = argc > argbase + 1 ? std::stoi(argv[argbase + 1]) : 20;
  size_t bytes = argc > argbase + 2 ? std::stoull(argv[argbase + 2]) : 25690112ULL;

  ucp_params_t params{};
  params.field_mask = UCP_PARAM_FIELD_FEATURES | UCP_PARAM_FIELD_REQUEST_SIZE;
  params.features = UCP_FEATURE_TAG;
  params.request_size = sizeof(ReqCtx);

  ucp_config_t* config = nullptr;
  check(ucp_config_read(nullptr, nullptr, &config), "ucp_config_read");

  ucp_context_h ctx = nullptr;
  check(ucp_init(&params, config, &ctx), "ucp_init");
  ucp_config_release(config);

  ucp_worker_params_t wparams{};
  wparams.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
  wparams.thread_mode = UCS_THREAD_MODE_SINGLE;

  ucp_worker_h worker = nullptr;
  check(ucp_worker_create(ctx, &wparams, &worker), "ucp_worker_create");

  ucp_address_t* local_addr = nullptr;
  size_t local_len = 0;
  check(ucp_worker_get_address(worker, &local_addr, &local_len), "ucp_worker_get_address");

  int fd = is_server ? listen_one(port) : connect_to(host, port);

  std::vector<char> peer_addr;
  exchange_addr(fd, local_addr, local_len, peer_addr);
  ucp_worker_release_address(worker, local_addr);

  ucp_ep_params_t ep_params{};
  ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
  ep_params.address = (ucp_address_t*)peer_addr.data();

  ucp_ep_h ep = nullptr;
  check(ucp_ep_create(worker, &ep_params, &ep), "ucp_ep_create");

  std::vector<char> buf(bytes);
  if (is_client) std::memset(buf.data(), 7, buf.size());

  char ready = 0;
  if (is_server) {
    send_all(fd, &ready, 1);
  } else {
    recv_all(fd, &ready, 1);
  }

  const uint64_t tag = 0xABCDEF;
  auto t0 = std::chrono::steady_clock::now();

  for (int i = 0; i < iters; ++i) {
    ReqCtx req_ctx{};
    void* req = nullptr;

    if (is_server) {
      req = ucp_tag_recv_nb(worker, buf.data(), bytes, ucp_dt_make_contig(1),
                            tag + i, UINT64_MAX, recv_cb);
    } else {
      req = ucp_tag_send_nb(ep, buf.data(), bytes, ucp_dt_make_contig(1),
                            tag + i, send_cb);
    }

    if (req && !UCS_PTR_IS_ERR(req)) {
      ReqCtx* real_ctx = (ReqCtx*)req;
      real_ctx->done = false;
      real_ctx->status = UCS_OK;
      wait_req(worker, req, *real_ctx);
    } else {
      wait_req(worker, req, req_ctx);
    }
  }

  auto t1 = std::chrono::steady_clock::now();
  double sec = std::chrono::duration<double>(t1 - t0).count();
  double gb = (double)bytes * iters / 1e9;
  double gbps = gb / sec;
  double gbit = gbps * 8.0;

  std::cout << "[UCX " << role << "] bytes=" << bytes
            << " iters=" << iters
            << " time_sec=" << sec
            << " GB/s=" << gbps
            << " Gbit/s=" << gbit
            << "\n";

  close(fd);
  ucp_ep_destroy(ep);
  ucp_worker_destroy(worker);
  ucp_cleanup(ctx);
  return 0;
}
